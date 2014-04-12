# Configuration file for CSCTriggerPrimitives building.
# Slava Valuev May-2006.

import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonCSCTriggerPrimitives")

process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring("file:/data0/slava/test/muminus_pt50_CMSSW_3_9_0_pre1.root"),
    fileNames = cms.untracked.vstring("file:muminus_pt50_CMSSW_6_1_0_pre2.root")
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
#process.GlobalTag.globaltag = 'DESIGN_31X_V8::All'
process.GlobalTag.globaltag = 'MC_61_V1::All'

# magnetic field (do I need it?)
# ==============================
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# CSC Trigger Primitives configuration
# ====================================
#process.load("L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesConfig_cff")
#process.l1csctpconf.isMTCC = True
#process.l1csctpconf.isTMB07 = True
#process.l1csctpconf.alctParamMTCC2.alctNplanesHitPretrig = 3
#process.l1csctpconf.alctParamMTCC2.alctNplanesHitAccelPretrig = 3
#process.l1csctpconf.clctParam.clctNplanesHitPretrig = 3
#process.l1csctpconf.clctParam.clctHitPersist = 4
#process.load("L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesDBConfig_cff")
#process.prefer("l1csctpdbconfsrc")

# CSC Trigger Primitives
# ======================
process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
process.cscTriggerPrimitiveDigis.alctParam07.verbosity = 2
process.cscTriggerPrimitiveDigis.clctParam07.verbosity = 2
process.cscTriggerPrimitiveDigis.tmbParam.verbosity = 2

#- For cosmic data
# process.cscTriggerPrimitiveDigis.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
# process.cscTriggerPrimitiveDigis.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"
# process.cscTriggerPrimitiveDigis.alctParam.alctTrigMode = 2

# Auxiliary services
# ==================
# process.Timing = cms.Service("Timing")
# process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")
# process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions")
# process.Tracer = cms.Service("Tracer")

process.out = cms.OutputModule("PoolOutputModule",
    # fileName = cms.untracked.string("lcts.root"),
    # fileName = cms.untracked.string("/data0/slava/test/lcts_muminus_pt50_emul_CMSSW_3_9_0_pre1.root"),
    fileName = cms.untracked.string("lcts_muminus_pt50_emul_CMSSW_6_1_0_pre2.root"),
    outputCommands = cms.untracked.vstring("keep *", 
        "drop *_DaqSource_*_*")
)

# Scheduler path
# ==============
process.p = cms.Path(process.cscTriggerPrimitiveDigis)
process.ep = cms.EndPath(process.out)
