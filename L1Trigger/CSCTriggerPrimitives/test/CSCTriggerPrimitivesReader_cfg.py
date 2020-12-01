import FWCore.ParameterSet.Config as cms

# Hack to add "test" directory to the python path.
import sys, os
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'],
                                'src/L1Trigger/CSCTriggerPrimitives/test'))

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process("L1CSCTriggerPrimitivesReader", Run2_2018)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:lcts.root")
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# For LogTrace to take an effect, compile using
# > scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('lctreader'),
    files = cms.untracked.PSet(
        debug = cms.untracked.PSet(
            extension = cms.untracked.string('.txt'),
            lineLength = cms.untracked.int32(132),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('DEBUG')
        )
    )
)

process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_61_V1::All'

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('TPEHists.root')
)

process.load("L1Trigger.CSCTriggerPrimitives.CSCTriggerPrimitivesReader_cfi")
process.lctreader.debug = True

process.p = cms.Path(process.lctreader)
