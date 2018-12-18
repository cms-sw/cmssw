# Configuration file to unpack CSC digis, run Trigger Primitives emulator,
# and compare LCTs in the data with LCTs found by the emulator.
# Slava Valuev; October, 2006.

import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("ME0TPEmulator", eras.Phase2)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(10)
)

# Hack to add "test" directory to the python path.
import sys, os
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'],
                                'src/L1Trigger/ME0Trigger/test'))

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

process.load('L1Trigger/ME0Trigger/me0TriggerPseudoDigis_cfi')
process.me0TriggerPseudoDigis.info = 3


# ======
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("ME0L1Digi.root"),
    outputCommands = cms.untracked.vstring("keep *",
        "drop *_DaqSource_*_*")
)


# Scheduler path
# ==============
process.p = cms.Path(process.me0TriggerPseudoDigis)
process.out = cms.EndPath(process.output)
