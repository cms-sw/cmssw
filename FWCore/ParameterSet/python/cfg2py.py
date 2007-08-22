import FWCore.ParameterSet.Config as cms
from sys import argv

print "import FWCore.ParameterSet.Config as cms"

f = open(argv[1])
s = f.read()

process = cms.processFromString(s)
print process.dumpPython()

