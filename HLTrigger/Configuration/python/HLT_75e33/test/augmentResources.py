import json

input_file = 'Phase2Timing_resources.json'
output_file = 'Phase2Timing_resources_abs.json'

orig = json.load(open(input_file,'r'))

time_real_abs = 0.
time_thread_abs = 0.

events = orig['total']['events']
orig['resources'].append({'time_real_abs': 'real time abs'})
orig['resources'].append({'time_thread_abs': 'cpu time abs'})

for k in orig['modules']:
    if k['events'] > 0 and k['label'] != "other":
        k['time_real_abs'] = k['time_real']/k['events']*events
        k['time_thread_abs'] = k['time_thread']/k['events']*events
        time_real_abs += k['time_real_abs']
        time_thread_abs += k['time_thread_abs']
    else:
        k['time_real_abs'] = 0
        k['time_thread_abs'] = 0

orig['total']['time_real_abs'] = time_real_abs
orig['total']['time_thread_abs'] = time_thread_abs

json.dump(orig, open(output_file, 'w'), indent=2)

