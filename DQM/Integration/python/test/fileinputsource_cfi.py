import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(
#        "file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root",
#        "file:/afs/cern.ch/user/m/muell149/public/outputA_25nsFrozenMenuTrgRes.root"
#        "file:/afs/cern.ch/user/m/muell149/work/CMSSW_7_4_0/src/HLTrigger/Configuration/test/outputA_extraCollections.root"
        "file:/afs/cern.ch/user/m/muell149/public/25ns_ppMenu_xtraCollections.root"
#        "/store/data/Commissioning2014/Cosmics/RAW//v3/000/224/380/00000/E05943D1-1227-E411-BB8E-02163E00F0C4.root"
#    "/store/data/Commissioning2014/Cosmics/RAW/v3/000/224/380/00000/68FDADE5-1227-E411-8AA6-02163E00A10C.root"
        )
)
maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
#    input = cms.untracked.int32(1000)
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
