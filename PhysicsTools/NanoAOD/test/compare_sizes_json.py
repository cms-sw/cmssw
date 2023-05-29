#!/usr/bin/env python3
import json

from optparse import OptionParser
parser = OptionParser(usage="%prog [options] job1 job2 ...")
parser.add_option("-f", "--format", dest="fmt", default="txt", help="output format")
parser.add_option("-H", "--header", dest="header", action="store_true", default=False, help="print headers")
parser.add_option("--base", dest="base", default="{}.json,{}_timing_report.json", help="coma separated base name for size and timing files")
parser.add_option("--ref", dest="ref", default="ref/", help="path to the reference files")
(options, args) = parser.parse_args()

headers = [ 'Sample' , 'kb/ev' , 'ref kb/ev' , 'diff kb/ev' , 'ev/s/thd' , 'ref ev/s/thd' , 'diff rate' , 'mem/thd' , 'ref mem/thd' ]
start, sep, end = "", "\t", ""
if options.fmt == "md": 
    start, sep, end = "| ", " | ", " |"

def prow(x): 
    print(start + sep.join(x) + end)

first = True

size_pattern,timing_pattern = options.base.split(',')
for job in args:

    label = job
    size_json=size_pattern.format(job)
    size_ref_json=options.ref+'/'+size_json
    timing_json=timing_pattern.format(job)
    timing_ref_json=options.ref+'/'+timing_json

    try:

        jnew = json.load(open(size_json,'r'))
        jref = json.load(open(size_ref_json,'r'))
        size_new = jnew["trees"]["Events"]['allsize']/jnew["trees"]["Events"]['entries']
        size_ref = jref["trees"]["Events"]['allsize']/jref["trees"]["Events"]['entries']

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
            prow(headers)
            if options.fmt == "md": prow("---" for x in headers)
            first = False

        prow([ label, '%.3f' % size_new, '%.3f' % size_ref, '%.3f ( %+.1f%% )' % (size_new - size_ref,  (size_new-size_ref)/size_ref * 100 ),
               '%.2f'%rate_new, '%.2f'%rate_ref, '%+.1f%%'%((rate_new-rate_ref)/rate_ref*100), '%.3f'%(rmem_new/1000), '%.3f'%(rmem_ref/1000)  ])

    except IOError: # some file not existing
        #print(f'file {size_json}, {size_ref_json}, {timing_json} or {timing_ref_json} does not exist')
        pass
