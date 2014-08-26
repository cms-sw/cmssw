import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(
        "file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root",
        )
)
maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Parameters for runType
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
from dqmPythonTypes import *

options = VarParsing.VarParsing('analysis')

options.register('runtype',
         'cosmic_run',
        VarParsing.VarParsing.multiplicity.singleton,
        VarParsing.VarParsing.varType.string,
          "Type of Run in CMS")

options.register ('runkey',
          'cosmic_run',
          VarParsing.VarParsing.multiplicity.singleton,
          VarParsing.VarParsing.varType.string,
          "Run Keys of CMS")

options.parseArguments()

# Fix to allow scram to compile
#if len(sys.argv) > 1:
#  options.parseArguments()

runType = RunType(['pp_run','cosmic_run','hi_run','hpu_run'])
if not options.runkey.strip():
  options.runkey = 'cosmic_run'

runType.setRunType(options.runkey.strip())
