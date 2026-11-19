#!/usr/bin/env cmsRun
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.pfnInPath import pfnInPath

import argparse
import sys
parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test the MCFileSource3 input source')
parser.add_argument("--inputFiles", type=str, nargs='+',
                    default=[pfnInPath('IOMC/Input/data/TTbar_13TeV_TuneCUETP8M1_HepMC3.hepmc3')],
                    help="HepMC3 or HepMC2 files to read")
parser.add_argument("--outputFile", type=str, default="HepMC3_GEN.root", help="Name of the output file")
parser.add_argument("--maxEvents", type=int, default=-1, help="Number of events to read (-1 for all)")
parser.add_argument("--printEvent", action="store_true", help="Print the content of every event which is read.")
args = parser.parse_args()

process = cms.Process("GEN")

process.source = cms.Source("MCFileSource3",
	fileNames = cms.untracked.vstring(args.inputFiles),
	printEvent = cms.untracked.bool(args.printEvent)
)

process.maxEvents.input = args.maxEvents


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO = cms.untracked.PSet(limit = cms.untracked.int32(-1))

process.GEN = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string(args.outputFile)
)

process.outpath = cms.EndPath(process.GEN)
