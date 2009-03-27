# Configuration file to unpack CSC digis, run Trigger Primitives emulator,
# and compare LCTs in the data with LCTs found by the emulator.
# Slava Valuev; October, 2006.

import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCTPEmulator")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# Hack to add "test" directory to the python path.
import sys, os
sys.path.insert(0, os.path.join(os.environ['CMSSW_BASE'],
                                'src/L1Trigger/CSCTriggerPrimitives/test'))

#    source = NewEventStreamFileReader {
##	string fileName = "file:/tmp/slava/mtcc.00002571.A.testStorageManager_0.0.dat"
#	untracked vstring fileNames = {
#	    "file:/tmp/slava/mtcc.00004138.A.testStorageManager_0.0.dat"
#	}
#        int32 max_event_size = 7000000
#        int32 max_queue_depth = 5
#    }

#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('file:/data0/slava/data/run58731/4C6067C2-B972-DD11-9672-000423D996B4.root')
##        untracked uint32 debugVebosity = 10
##        untracked bool   debugFlag     = false
###	untracked uint32 skipEvents    = 2370
#)
process.load("localrun_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring("debug"),
    debug = cms.untracked.PSet(
        extension = cms.untracked.string(".txt"),
        threshold = cms.untracked.string("DEBUG"),
        # threshold = cms.untracked.string("WARNING"),
        lineLength = cms.untracked.int32(132),
        noLineBreaks = cms.untracked.bool(True)
    ),
    # debugModules = cms.untracked.vstring("*")
    debugModules = cms.untracked.vstring("cscTriggerPrimitiveDigis", 
        "lctreader")
)

# es_source of ideal geometry
# ===========================
#process.load("Configuration/StandardSequences/Geometry_cff")

# endcap muon only...
process.load("Geometry.MuonCommonData.muonEndcapIdealGeometryXML_cfi")

# Needed according to Mike Case's e-mail from 27/03.
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

# flags for modelling of CSC geometry
# ===================================
process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'STARTUP_30X::All'
#process.prefer("GlobalTag")

# magnetic field (do I need it?)
# ==============================
process.load("Configuration.StandardSequences.MagneticField_cff")

# Cabling map
# ===========
#process.load("EventFilter.CSCRawToDigi.cscFrontierCablingUnpck_cff")

# CSC raw --> digi unpacker
# =========================
process.cscunpacker = cms.EDFilter("CSCDCCUnpacker",
    Debug = cms.untracked.bool(False),
    PrintEventNumber = cms.untracked.bool(False),
    UseExaminer = cms.untracked.bool(True),
    # ExaminerMask = cms.untracked.uint32(0x7FF7BF6),
    ExaminerMask = cms.untracked.uint32(0x1FEBF3F6),
    UnpackStatusDigis = cms.untracked.bool(False),
    # for run 566 and 2008 data
    ErrorMask = cms.untracked.uint32(0),
    # ErrorMask = cms.untracked.uint32(0xDFCFEFFF),
    UseSelectiveUnpacking = cms.untracked.bool(True),
    # Define input to the unpacker
    # InputObjects = cms.InputTag("cscpacker","CSCRawData"),
    InputObjects = cms.InputTag("source"),
    # MTCC data flag
    isMTCCData = cms.untracked.bool(False)
)

# CSC Trigger Primitives configuration
# ====================================
#process.load("L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesConfig_cff")
#process.load("L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesDBConfig_cff")
#process.prefer("l1csctpdbconfsrc")
# process.l1csctpconf.alctParamMTCC2.alctDriftDelay = 2
# process.l1csctpconf.alctParamMTCC2.alctL1aWindowWidth = 7
#process.l1csctpconf.alctParamMTCC2.alctDriftDelay = 9
#process.l1csctpconf.alctParamMTCC2.alctL1aWindowWidth = 9


# CSC Trigger Primitives emulator
# ===============================
process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
process.cscTriggerPrimitiveDigis.alctParam07.verbosity = 2
process.cscTriggerPrimitiveDigis.clctParam07.verbosity = 2
process.cscTriggerPrimitiveDigis.tmbParam.verbosity = 2
process.cscTriggerPrimitiveDigis.CSCComparatorDigiProducer = "cscunpacker:MuonCSCComparatorDigi"
process.cscTriggerPrimitiveDigis.CSCWireDigiProducer = "cscunpacker:MuonCSCWireDigi"

# CSC Trigger Primitives reader
# =============================
process.load("CSCTriggerPrimitivesReader_cfi")
process.lctreader.debug = True
process.lctreader.isMTCCData = False

#process.myfilter = cms.EDFilter(
#  'EventNumberFilter',
#  runEventNumbers = cms.vuint32(1,4309, 1,4310)
#)

# Output
# ======
process.out = cms.OutputModule("PoolOutputModule",
    #fileName = cms.untracked.string("/data0/slava/test/lcts_run58731.root"),
    #fileName = cms.untracked.string("/data0/slava/test/lcts_run59318.root"),
    fileName = cms.untracked.string("/data0/slava/test/lcts_run62232.root"),
    #fileName = cms.untracked.string("/data0/slava/test/lcts_run65882.root"),
    outputCommands = cms.untracked.vstring("keep *", 
        "drop *_DaqSource_*_*")
)

# Scheduler path
# ==============
#process.p = cms.Path(process.myfilter*process.cscunpacker*process.cscTriggerPrimitiveDigis*process.lctreader)
process.p = cms.Path(process.cscunpacker*process.cscTriggerPrimitiveDigis*process.lctreader)
process.ep = cms.EndPath(process.out)
