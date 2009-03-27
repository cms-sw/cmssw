import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripDQMFile")

process.MessageLogger = cms.Service(
    "MessageLogger",
    debugModules = cms.untracked.vstring('SiStripMonitorTrack'), 
                                         
    debug = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('debug')
    )

#-------------------------------------------------
# Magnetic Field
#-------------------------------------------------
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.prefer("VolumeBasedMagneticFieldESProducer")


#-------------------------------------------------
# CMS Geometry
#-------------------------------------------------
process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_ALL_V4::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#process.sistripconn = cms.ESProducer("SiStripConnectivity")

#-------------------------------------------------
# DQM
#-------------------------------------------------
# DQMStore Service
process.DQMStore = cms.Service("DQMStore",
                               referenceFileName = cms.untracked.string(''),
                               verbose = cms.untracked.int32(0)
                               )
# SiStripMonitorTrack
process.load("DQM.SiStripMonitorTrack.SiStripMonitorTrack_StandAlone_cff")

#-------------------------------------------------
# Performance Checks
#-------------------------------------------------
# memory
#process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
#                                        ignoreTotal = cms.untracked.int32(0)
#                                        )

# timing
#process.Timing = cms.Service("Timing")

#-------------------------------------------------
# In-/Output
#-------------------------------------------------

# input
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(

    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/2EA98EBE-07C2-DD11-9584-001D0967D0DF.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/EA0B6FEA-AFC1-DD11-A9CA-001D0967BC3E.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0007/D08B578B-7CC4-DD11-9754-001D0967D24C.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0051/2A2DAF71-E6CB-DD11-9DF3-001D0967D5FD.root'
     
    )
                            )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

# output
#process.out = cms.OutputModule("PoolOutputModule",
#                               fileName = cms.untracked.string('TESTReal.root'),
#                               options = cms.PSet(wantSummary = cms.untracked.bool(True))                               
#                               )

#-------------------------------------------------
# Scheduling
#-------------------------------------------------
#process.dumpinfo = cms.EDAnalyzer("EventContentAnalyzer")

process.outP = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.DQMSiStripMonitorTrack_Real)
process.pout = cms.EndPath(process.outP)
