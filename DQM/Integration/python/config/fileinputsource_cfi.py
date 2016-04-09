import FWCore.ParameterSet.Config as cms

source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/5669AE2E-CAFC-E511-9402-02163E01452B.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/266/681/00000/A459B232-CAFC-E511-B119-02163E011856.root",
        "file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root",
        #"/store/data/Commissioning2014/Cosmics/RAW//v3/000/224/380/00000/E05943D1-1227-E411-BB8E-02163E00F0C4.root"
        #"/store/data/Commissioning2014/Cosmics/RAW/v3/000/224/380/00000/68FDADE5-1227-E411-8AA6-02163E00A10C.root"
    )
 
    # For .dat files, use the following block instead.   
    #"NewEventStreamFileReader",
    #fileNames = cms.untracked.vstring(
    #    "/store/t0streamer/Data/ExpressCosmics/000/267/410/run267410_ls0001_streamExpressCosmics_StorageManager.dat",
    #)
)

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Parameters for runType
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
from dqmPythonTypes import *

options = VarParsing.VarParsing("analysis")

options.register(
    "runkey",
    "cosmic_run",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Run Keys of CMS"
)

options.parseArguments()

# Fix to allow scram to compile
#if len(sys.argv) > 1:
#  options.parseArguments()

runType = RunType()
if not options.runkey.strip():
    options.runkey = "pp_run"

runType.setRunType(options.runkey.strip())
