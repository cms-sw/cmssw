import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

# Hack to add "test" directory to the python path.
import sys, os
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'],
                                'src/L1Trigger/CSCTriggerPrimitives/test'))

process = cms.Process("L1CSCTriggerPrimitivesReader", eras.Run2_2018)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:lcts.root")
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# For LogTrace to take an effect, compile using
# > scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring("debug"),
    #	untracked vstring categories     = { "lctDigis" }
    #	untracked vstring debugModules   = { "*" }
    #	untracked PSet debugmessages.txt = {
    #	    untracked string threshold = "DEBUG"
    #	    untracked PSet INFO     = {untracked int32 limit = 0}
    #	    untracked PSet DEBUG    = {untracked int32 limit = 0}
    #	    untracked PSet lctDigis = {untracked int32 limit = 10000000}
    #	}
    debug = cms.untracked.PSet(
        threshold = cms.untracked.string("DEBUG"),
        extension = cms.untracked.string(".txt"),
        lineLength = cms.untracked.int32(132),
        noLineBreaks = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring("lctreader")
)

process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_61_V1::All'

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('TPEHists.root')
)

process.load("CSCTriggerPrimitivesReader_cfi")
process.lctreader.debug = True

process.p = cms.Path(process.lctreader)
