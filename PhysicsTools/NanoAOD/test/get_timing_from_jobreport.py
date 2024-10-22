#!/usr/bin/env python3
import sys
import xml.etree.ElementTree
import json
import os

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("input", metavar="input.xml", type=str)
parser.add_argument("output", metavar="output.json", type=str)
options = parser.parse_args()

timing={}

try:
    tree = xml.etree.ElementTree.parse(options.input)
    timing['NumberEvents']=int(tree.find("./PerformanceReport/PerformanceSummary[@Metric='ProcessingSummary']/Metric[@Name='NumberEvents']").get('Value'))
    for x in tree.findall("./PerformanceReport/PerformanceSummary[@Metric='Timing']/Metric"):
        timing['Timing/'+x.get('Name')]=float(x.get('Value'))
    for x in tree.findall("./PerformanceReport/PerformanceSummary[@Metric='ApplicationMemory']/Metric"):
        timing['ApplicationMemory/'+x.get('Name')]=float(x.get('Value'))
except Exception as e:
    print("Could not parse job report %s, content is:" % options.input)
    os.system("cat %s" % options.input)
    raise e


with open(options.output,'w') as f:
    json.dump(timing,f)
