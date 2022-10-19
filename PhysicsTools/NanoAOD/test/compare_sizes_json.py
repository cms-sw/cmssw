#!/usr/bin/env python3
import json

from optparse import OptionParser
parser = OptionParser(usage="%prog [options] job1 job2 ...")
parser.add_option("-f", "--format", dest="fmt", default="txt", help="output format")
parser.add_option("-H", "--header", dest="header", action="store_true", default=False, help="print headers")
(options, args) = parser.parse_args()

headers = [ 'Sample' , 'kb/ev' , 'ref kb/ev' , 'diff kb/ev' , 'ev/s/thd' , 'ref ev/s/thd' , 'diff rate' , 'mem/thd' , 'ref mem/thd' ]
start, sep, end = "", "\t", ""
if options.fmt == "md": 
    start, sep, end = "| ", " | ", " |"

def prow(x): 
    print(start + sep.join(x) + end)

first = True

for job in args:

    label = job
    me = '%s.json'%job
    ref = 'ref/%s'%me
    me_t = '%s_timing_report.json'%job
    ref_t = 'ref/%s'%me_t

    try:

        jnew = json.load(open(me,'r'))
        jref = json.load(open(ref,'r'))
        size_new = jnew["trees"]["Events"]['allsize']/jnew["trees"]["Events"]['entries']
        size_ref = jref["trees"]["Events"]['allsize']/jref["trees"]["Events"]['entries']

        jnew_t = json.load(open(me_t,'r'))
        jref_t = json.load(open(ref_t,'r'))
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
        pass
