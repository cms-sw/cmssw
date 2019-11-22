#!/usr/bin/env python
from __future__ import print_function

import uproot
import argparse

from prettytable import PrettyTable
from collections import defaultdict

parser = argparse.ArgumentParser(description="Shows Indices table in a DQMIO file. Last column (ME count) is computed like this: lastIndex - firstIndex + 1")

parser.add_argument('filename', help='Name of local root file. For remote files, use edmCopyUtil first: `edmCopyUtil root://cms-xrd-global.cern.ch/<FILEPATH> .`')

args = parser.parse_args()

typeNames = ['Ints','Floats', 'Strings', 'TH1Fs','TH1Ss', 'TH1Ds',
             'TH2Fs', 'TH2Ss', 'TH2Ds', 'TH3Fs', 'TProfiles','TProfile2Ds']

f = uproot.open(args.filename)
things = f.keys()
if 'Indices;1' in things:
  indices = f['Indices']
  runs = indices.array('Run')
  lumis = indices.array('Lumi')
  firstindex = indices.array('FirstIndex')
  lastindex = indices.array('LastIndex')
  types = indices.array('Type')

  table = PrettyTable()
  table.field_names = ['Run', 'Lumi', 'FirstIndex', 'LastIndex', 'Type', 'ME Count']
  
  for run, lumi, first, last, type in zip(runs, lumis, firstindex, lastindex, types):
    typeName = 'Unknown'
    if type < len(typeNames):
      typeName = typeNames[type]
    
    table.add_row([run, lumi, first, last, '%s (%s)' % (type, typeName), int(last - first + 1)])

  print(table)
else:
  print("This does not look like DQMIO data.")
