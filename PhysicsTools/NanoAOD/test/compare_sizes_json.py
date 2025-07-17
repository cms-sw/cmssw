#!/usr/bin/env python3
import json

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--format", dest="fmt", default="txt", help="output format")
parser.add_argument("-H", "--header", dest="header", action="store_true", default=False, help="print headers")
parser.add_argument("--base", dest="base", default="{}.json,{}_timing_report.json", help="comma-separated base name for size and timing files")
parser.add_argument("--ref", dest="ref", default="ref/", help="path to the reference files")
parser.add_argument("job", type=str, nargs='+')
options = parser.parse_args()

headers = [ 'workflow' , 'id' , 'kb/ev' , 'ref kb/ev' , 'diff kb/ev' , 'ev/s/thd' , 'ref ev/s/thd' , 'diff rate' , 'mem/thd' , 'ref mem/thd' ]
start, sep, end = "", "\t", ""
if options.fmt == "md": 
    start, sep, end = "| ", " | ", " |"

def prow(x): 
    print(start + sep.join(x) + end)

first = True

size_pattern,timing_pattern = options.base.split(',')
for job in options.job:

    try:
        wf_id = job.split("_")[0]
        wf_name = "_".join(job.split("_")[1:])
    # Just in case... Should never happen.
    except IndexError:
        wf_id = ""
        wf_name = job

    size_json=size_pattern.format(job)
    size_ref_json=options.ref+'/'+size_json
    timing_json=timing_pattern.format(job)
    timing_ref_json=options.ref+'/'+timing_json

    try:

        jnew = json.load(open(size_json,'r'))
        jref = json.load(open(size_ref_json,'r'))

        entries_new = jnew["trees"]["Events"]['entries']
        entries_ref = jref["trees"]["Events"]['entries']

        size_new = jnew["trees"]["Events"]['allsize']/entries_new if entries_new > 0.0 else 0.0
        size_ref = jref["trees"]["Events"]['allsize']/entries_ref if entries_ref > 0.0 else 0.0

        rel_diff = (size_new-size_ref)/size_ref * 100 if size_ref !=  0.0 else 0.0

        jnew_t = json.load(open(timing_json,'r'))
        jref_t = json.load(open(timing_ref_json,'r'))
        rate_new = jnew_t["Timing/EventThroughput"]/jnew_t["Timing/NumberOfThreads"]
        rate_ref = jref_t["Timing/EventThroughput"]/jref_t["Timing/NumberOfThreads"]
        try:
            rmem_new = jnew_t["ApplicationMemory/PeakValueRss"]/jnew_t["Timing/NumberOfThreads"]
            rmem_ref = jref_t["ApplicationMemory/PeakValueRss"]/jref_t["Timing/NumberOfThreads"]
        except KeyError:
            rmem_new = 0
            rmem_ref = 0

        if first and options.header:
            print("\n")
            prow(headers)
            if options.fmt == "md": prow("---" for x in headers)
            first = False

        prow([ wf_name, wf_id, '%.3f' % size_new, '%.3f' % size_ref, '%.3f ( %+.1f%% )' % (size_new - size_ref,  rel_diff ),
               '%.2f'%rate_new, '%.2f'%rate_ref, '%+.1f%%'%((rate_new-rate_ref)/rate_ref*100), '%.3f'%(rmem_new/1000), '%.3f'%(rmem_ref/1000)  ])

    except IOError: # some file not existing
        #print(f'file {size_json}, {size_ref_json}, {timing_json} or {timing_ref_json} does not exist')
        pass
