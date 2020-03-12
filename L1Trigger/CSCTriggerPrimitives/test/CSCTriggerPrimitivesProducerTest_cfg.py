# Configuration file for CSCTriggerPrimitives building.
# Slava Valuev May-2006.

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process("MuonCSCTriggerPrimitives", Run2_2018)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("digis.root")
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring("log", "debug", "errors"),
    statistics = cms.untracked.vstring("stat"),
    # No constraint on log.txt content...
    log = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        lineLength = cms.untracked.int32(132),
        noLineBreaks = cms.untracked.bool(True)
    ),
    debug = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        threshold = cms.untracked.string("DEBUG"),
        lineLength = cms.untracked.int32(132),
        noLineBreaks = cms.untracked.bool(True)
    ),
    errors = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        threshold = cms.untracked.string("ERROR")
    ),
    stat = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        threshold = cms.untracked.string("INFO")
    ),
    # turn on the following to get LogDebug output
    # ============================================
    # debugModules = cms.untracked.vstring("*"),
    debugModules = cms.untracked.vstring("cscTriggerPrimitiveDigis")
)

# es_source of ideal geometry
# ===========================
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_61_V1::All'

# magnetic field (do I need it?)
# ==============================
process.load("Configuration.StandardSequences.MagneticField_cff")

# CSC Trigger Primitives configuration
# ====================================
#process.load("L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesConfig_cff")
#process.load("L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesDBConfig_cff")

# CSC Trigger Primitives
# ======================
process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
process.cscTriggerPrimitiveDigis.alctParam07.verbosity = 2
process.cscTriggerPrimitiveDigis.clctParam07.verbosity = 2
process.cscTriggerPrimitiveDigis.tmbParam.verbosity = 2

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("out_lcts.root"),
    outputCommands = cms.untracked.vstring("keep *",
        "drop *_DaqSource_*_*")
)

# Scheduler path
# ==============
process.p = cms.Path(process.cscTriggerPrimitiveDigis)
process.ep = cms.EndPath(process.out)
