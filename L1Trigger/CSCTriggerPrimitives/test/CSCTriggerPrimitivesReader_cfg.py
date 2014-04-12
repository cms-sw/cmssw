import FWCore.ParameterSet.Config as cms

# Hack to add "test" directory to the python path.
import sys, os
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'],
                                'src/L1Trigger/CSCTriggerPrimitives/test'))

process = cms.Process("L1CSCTriggerPrimitivesReader")

process.source = cms.Source("PoolSource",
    # fileNames = cms.untracked.vstring("file:lcts.root"),
    # fileNames = cms.untracked.vstring("file:/data0/slava/test/lcts_muminus_pt50_emul_CMSSW_3_9_0_pre1.root"),
    fileNames = cms.untracked.vstring("file:lcts_muminus_pt50_emul_CMSSW_6_1_0_pre2.root")
)
#process.PoolSource.fileNames = ['/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0004/EE15A7EC-E641-DE11-A279-001D09F29321.root']

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
#process.GlobalTag.globaltag = 'DESIGN_31X_V8::All'
process.GlobalTag.globaltag = 'MC_61_V1::All'

# Enable floating point exceptions
#process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions")
#process.Tracer = cms.Service("Tracer")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('TPEHists.root')
)

process.load("CSCTriggerPrimitivesReader_cfi")
process.lctreader.debug = True
process.lctreader.dataLctsIn = False
#- For official RelVal samples
#process.lctreader.CSCLCTProducerEmul = "simCscTriggerPrimitiveDigis"

process.p = cms.Path(process.lctreader)
