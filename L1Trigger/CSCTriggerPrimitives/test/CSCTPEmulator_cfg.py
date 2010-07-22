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

process.source = cms.Source("PoolSource",
##    fileNames = cms.untracked.vstring('file:/data0/slava/data/run58731/4C6067C2-B972-DD11-9672-000423D996B4.root')
#     fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW/v1/000/069/912/007F93B1-18AD-DD11-AB6D-000423D991D4.root'),
     fileNames = cms.untracked.vstring(
#       '/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/FE316E49-047F-DE11-AC0C-001D09F231B0.root'
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/FCEFF16D-F57E-DE11-B5D6-0030487A1FEC.root',
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/FC9CDB54-FF7E-DE11-B93F-001D09F24024.root',
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/F8DAC763-067F-DE11-8539-001D09F295A1.root',
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/F847EBEF-F87E-DE11-AAC5-000423D94A20.root',
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/F6240161-EC7E-DE11-B347-000423D952C0.root',
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/F4CCFBD1-077F-DE11-BAA9-0030487A18F2.root',
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/F405FCD9-077F-DE11-A59E-000423D944FC.root',
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/F2FA627F-087F-DE11-8931-001D09F2AD4D.root',
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/F2F426F5-1A7F-DE11-8930-001D09F24303.root',
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/F2E6A530-F67E-DE11-8B19-001D09F248F8.root',
#       'rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW/v1/000/109/562/F2D0CA18-187F-DE11-B296-001D09F2545B.root'
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/122/909/F81EA88E-A5DB-DE11-AA71-00304879FBB2.root'
     )
#     fileNames = cms.untracked.vstring('file:/data0/slava/data/run109562/FE316E49-047F-DE11-AC0C-001D09F231B0.root')
##        untracked uint32 debugVebosity = 10
##        untracked bool   debugFlag     = false
###	untracked uint32 skipEvents    = 2370
)
#process.load("localrun_cfi")

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
process.GlobalTag.globaltag = 'MC_3XY_V21::All'
#process.GlobalTag.globaltag = 'CRAFT09_R_V9::All'
#process.prefer("GlobalTag")

# magnetic field (do I need it?)
# ==============================
process.load("Configuration.StandardSequences.MagneticField_cff")

# CSC raw --> digi unpacker
# =========================
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
# InputObjects = cms.InputTag("cscpacker","CSCRawData")
# for run 566 and 2008 data
# ErrorMask = cms.untracked.uint32(0xDFCFEFFF)

# CSC Trigger Primitives configuration
# ====================================
#process.load("L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesConfig_cff")
#process.load("L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesDBConfig_cff")
#process.prefer("l1csctpdbconfsrc")
#process.l1csctpconf.alctParamMTCC2.alctNplanesHitPretrig = 3
#process.l1csctpconf.alctParamMTCC2.alctNplanesHitAccelPretrig = 3
#process.l1csctpconf.clctParam.clctNplanesHitPretrig = 3
#process.l1csctpconf.clctParam.clctHitPersist = 4
#process.l1csctpconf.alctParamMTCC2.alctDriftDelay = 9
#process.l1csctpconf.alctParamMTCC2.alctL1aWindowWidth = 9

# CSC Trigger Primitives emulator
# ===============================
process.load("L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi")
process.cscTriggerPrimitiveDigis.alctParam07.verbosity = 2
process.cscTriggerPrimitiveDigis.clctParam07.verbosity = 2
process.cscTriggerPrimitiveDigis.tmbParam.verbosity = 2
process.cscTriggerPrimitiveDigis.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
process.cscTriggerPrimitiveDigis.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"

# CSC Trigger Primitives reader
# =============================
process.load("CSCTriggerPrimitivesReader_cfi")
process.lctreader.debug = True

# Auxiliary services
# ==================
#process.SimpleProfiling = cms.Service("SimpleProfiling")

#process.myfilter = cms.EDFilter(
#  'EventNumberFilter',
#  runEventNumbers = cms.vuint32(1,4309, 1,4310)
#)

# Output
# ======
process.out = cms.OutputModule("PoolOutputModule",
    #fileName = cms.untracked.string("/data0/slava/test/lcts_run122909.root"),
    fileName = cms.untracked.string("lcts_run122909.root"),
    outputCommands = cms.untracked.vstring("keep *", 
        "drop *_DaqSource_*_*")
)

# Scheduler path
# ==============
#process.p = cms.Path(process.myfilter*process.muonCSCDigis*process.cscTriggerPrimitiveDigis*process.lctreader)
process.p = cms.Path(process.muonCSCDigis*process.cscTriggerPrimitiveDigis*process.lctreader)
process.ep = cms.EndPath(process.out)
