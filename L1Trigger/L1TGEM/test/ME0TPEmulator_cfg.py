import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2_cff import Phase2
process = cms.Process("ME0TPEmulator", Phase2)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(10)
)


process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
         'file:ME0_step3.root'
     )
)

# For LogTrace to take an effect, compile using
# > scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring("debug"),
    debug = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        threshold = cms.untracked.string("DEBUG"),
        # threshold = cms.untracked.string("WARNING"),
        lineLength = cms.untracked.int32(132),
        noLineBreaks = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring("me0TriggerPseudoDigis")
)

# es_source of ideal geometry
# ===========================
#process.load("Configuration/StandardSequences/GeometryDB_cff")
process.load('Configuration.Geometry.GeometryExtended2023D22Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.load('L1Trigger.L1TGEM.me0TriggerConvertedPseudoDigis_cfi')
process.me0TriggerConvertedPseudoDigis.info = 3


# ======
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("ME0L1Digi.root"),
    outputCommands = cms.untracked.vstring("keep *",
        "drop *_DaqSource_*_*")
)


# Scheduler path
# ==============
process.p = cms.Path(process.me0TriggerConvertedPseudoDigis)
process.out = cms.EndPath(process.output)
