import FWCore.ParameterSet.Config as cms
import sys

print(sys.argv)

process = cms.Process("TEST")
process.source = cms.Source("EmptySource")

process.maxEvents.input = 1
