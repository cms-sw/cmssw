import FWCore.ParameterSet.Config as cms

# Hack to add "test" directory to the python path.
import sys, os
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'],
                                'src/L1Trigger/CSCTriggerPrimitives/test'))

process = cms.Process("L1CSCTriggerPrimitivesReader")

process.source = cms.Source("PoolSource",
    # fileNames = cms.untracked.vstring("file:lcts.root"),
    fileNames = cms.untracked.vstring("file:/data0/slava/test/lcts_muminus_pt50_emul_CMSSW_3_1_0_pre4.root.blockedME1A"),
    # fileNames = cms.untracked.vstring("file:/data0/slava/test/lcts_14419l.root.sav"),
    debugVebosity = cms.untracked.uint32(10),
    debugFlag = cms.untracked.bool(False)
)
#process.PoolSource.fileNames = ["/store/mc/2007/8/10/RelVal-RelVal160pre9SingleMuPlusPt100L1-CMSSW_1_6_0_pre9-1186773613/0000/242BFAAC-E847-DC11-9B17-001731AF6701.root"]

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

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

process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_30X::All'

# Enable floating point exceptions
#process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions")
#process.Tracer = cms.Service("Tracer")

process.load("CSCTriggerPrimitivesReader_cfi")
process.lctreader.debug = True
process.lctreader.dataLctsIn = False

process.p = cms.Path(process.lctreader)
