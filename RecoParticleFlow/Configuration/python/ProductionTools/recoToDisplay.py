#!/usr/bin/env python
# Colin Bernet, Dec 2009

import FWCore.ParameterSet.Config as cms
from optparse import OptionParser

import sys,os, re, pprint, imp


parser = OptionParser()
parser.usage = "%prog <run> <lumi> <event> <cfg>"

(options,args) = parser.parse_args()

if len(args) != 4:
    parser.print_help()
    sys.exit(1)

run = args[0]
lumi = args[1]
event = args[2]
cfg = args[3]

files = os.popen('dbs search --query "find file where dataset=/ExpressPhysics/BeamCommissioning09-Express-v2/FEVT and run=%s and lumi=%s"' % (run, lumi) )

for file in files:
    print file

handle = open(cfg, 'r')
cfo = imp.load_source("pycfg", cfg, handle)
process = cfo.process
handle.close()

print process.dumpPython()
