#!/usr/bin/env python3
import sys
import xml.etree.ElementTree
import json
import os

from optparse import OptionParser
parser = OptionParser(usage="%prog input.xml output.json")
(options, args) = parser.parse_args()
if len(args)!=2: raise RuntimeError

timing={}

try:
    tree = xml.etree.ElementTree.parse(args[0])
    timing['NumberEvents']=int(tree.find("./PerformanceReport/PerformanceSummary[@Metric='ProcessingSummary']/Metric[@Name='NumberEvents']").get('Value'))
    for x in tree.findall("./PerformanceReport/PerformanceSummary[@Metric='Timing']/Metric"):
        timing['Timing/'+x.get('Name')]=float(x.get('Value'))
    for x in tree.findall("./PerformanceReport/PerformanceSummary[@Metric='ApplicationMemory']/Metric"):
        timing['ApplicationMemory/'+x.get('Name')]=float(x.get('Value'))
except Exception as e:
    print("Could not parse job report %s, content is:" % args[0])
    os.system("cat %s" % args[0])
    raise e


with open(args[1],'w') as f:
    json.dump(timing,f)
