import FWCore.ParameterSet.Config as cms

source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        #"root://eoscms//store/express/Commissioning2016/ExpressPhysics/FEVT/Express-v1/000/269/612/00000/0CAC70DE-3C01-E611-84B8-02163E01346D.root", 
        #"root://eoscms//store/express/Commissioning2016/ExpressPhysics/FEVT/Express-v1/000/269/612/00000/0EC9F919-4001-E611-BB95-02163E01391D.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressPhysics/FEVT/Express-v1/000/269/612/00000/1CF4C821-3F01-E611-A25A-02163E012A04.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressPhysics/FEVT/Express-v1/000/269/612/00000/586CE231-3E01-E611-8D60-02163E0127E3.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressPhysics/FEVT/Express-v1/000/269/612/00000/5C448328-3E01-E611-92A4-02163E0142A8.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressPhysics/FEVT/Express-v1/000/269/612/00000/7EBF1912-3F01-E611-B2F3-02163E01339E.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressPhysics/FEVT/Express-v1/000/269/612/00000/8A77C6E0-3B01-E611-AC71-02163E014557.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressPhysics/FEVT/Express-v1/000/269/612/00000/94D3D8D5-3C01-E611-BE3A-02163E01391D.root",
        #"/store/data/Commissioning2016/ZeroBias1/RAW/v1/000/269/060/00000/06A51B35-27FF-E511-9155-02163E0144C3.root",
        #"/store/data/Commissioning2016/ZeroBias1/RAW/v1/000/269/060/00000/0AEFA3F4-1DFF-E511-914E-02163E011B3F.root",
        #"/store/data/Commissioning2016/ZeroBias1/RAW/v1/000/269/060/00000/1E91A739-1AFF-E511-A3CE-02163E011F71.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/268/733/00000/5669AE2E-CAFC-E511-9402-02163E01452B.root",
        #"root://eoscms//store/express/Commissioning2016/ExpressCosmics/FEVT/Express-v1/000/268/733/00000/A459B232-CAFC-E511-B119-02163E011856.root",
        "file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root",
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
