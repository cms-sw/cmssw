#!/usr/bin/env python3
import json

class SlotManager(object):
    def __init__(self):
        self._endTimeForSlots: [int] = [0]
    def findSlot(self, startTime:int, endTime:int) -> int:
        for i,lastEndTime in enumerate(self._endTimeForSlots):
            if lastEndTime < startTime:
                self._endTimeForSlots[i] = endTime
                return i+1
        index = len(self._endTimeForSlots)
        self._endTimeForSlots.append(endTime)
        return index+1
    

def parseJson(input, field):
    traceEvents = [{"name":"program start", "ph":"i", "ts":0, "pid":1, "tid":1}]

    slotManager = SlotManager()
    for m in input['measurements']:
        start = m['timeRange'][0]
        end = m['timeRange'][1]
        slot = slotManager.findSlot(start, end)
        value = m['alloc'][field]
        absValue = value
        sign = 'pos'
        if value < 0:
            absValue = -1*value
            sign = 'neg'
        threshold:int = 0
        categories = m['label']+','+m['activity']+','+m['transition']
        while absValue >= threshold:
            if threshold == 0:
                name=m['label']
                threshold = 1
            else:
                name=sign
                threshold *=10
            traceEvents.append(dict(name=name, cat=categories, ph ='X', pid=1, tid=slot, ts=start, dur=end-start, args={field:value}))
    trace= dict(traceEvents=traceEvents)
    return trace   
                
    
if __name__=="__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Convert time based json file from edmModuleAllocAnalyze.py into chrome tracer format.')

    parser.add_argument('filename',
                        type=argparse.FileType('r'),
                        help='file to process')
    parser.add_argument('-f','--field',
                        type=str,
                        default='added',
                        help='''Specify which of the field values to histogram. The allowed options are
 'added': how much extra memory was added on that thread during that period [default].
 'nAlloc': number of allocations done on that thread during that period
 'nDealloc : number of deallocations done on that thread during that period
 'max1Alloc' : largest allocation request done on that thread during that period
 'minTemp' : minimum sum of allocations - deallocations requested memory done on that thread during that period (can be negative)
 'maxTemp' : maximum sum of allocations - deallocations requested memory done on that thread during that period.''')

    args = parser.parse_args()
    if args.field not in ['added', 'nAlloc', 'nDealloc', 'max1Alloc', 'minTemp', 'maxTemp']:
        raise RuntimeError(f"unknown field request of '{args.field}'")

    input = json.load(args.filename)
    result = parseJson(input, args.field)
    print(json.dumps(result, indent=2))
