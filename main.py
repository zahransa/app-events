import os
import numpy as np
import mne
import json
import helper
from mne_bids import BIDSPath, write_raw_bids
import shutil
import matplotlib.pyplot as plt

#workaround for -- _tkinter.TclError: invalid command name ".!canvas"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('config.json') as config_json:
    config = json.load(config_json)

data_file = config['fif']
raw = mne.io.read_raw_fif(data_file, verbose=False)

# Create a BIDSPath
bids_path = BIDSPath(subject='subject',
                     session=None,
                     task='task',
                     run='01',
                     acquisition=None,
                     processing=None,
                     recording=None,
                     space=None,
                     suffix=None,
                     datatype='meg',
                     root='bids')



events = mne.find_events(raw, stim_channel=config['stim_channel'],
                             consecutive=config['consecutive'], mask=config['mask'],
                             mask_type=config['mask_type'], min_duration=config['min_duration'])
report = mne.Report(title='Event')

sfreq = raw.info['sfreq']



events = mne.pick_events(events, exclude=config['exclude'])

events = mne.merge_events(events, ids=config['ids'], new_id=config['new_id'])


event_id_condition= config['event_id_condition']
# Convert String to Dictionary using strip() and split() methods
event_id = dict((x.strip(), int(y.strip()))
                for x, y in (element.split('-')
                             for element in event_id_condition.split(', ')))

id_list = list(event_id.values())

events = mne.pick_events(events, include=id_list)



# # Write BIDS to create events.tsv BIDS compliant
write_raw_bids(raw, bids_path, events_data=events, event_id=event_id, overwrite=True)
#
# # Extract events.tsv from bids path
events_file = 'bids/sub-subject/meg/sub-subject_task-task_run-01_events.tsv'
#
# # Copy events.tsv in outdir
shutil.copy2(events_file, 'out_dir/events.tsv')





# == FIGURES ==
plt.figure(1)
fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp, event_id=event_id)
fig.subplots_adjust(right=0.7)  # make room for legend
fig.savefig(os.path.join('out_figs','events.png'))
