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
process.load("Configuration.StandardSequences.MagneticField_0T_cff")
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
process.GlobalTag.globaltag = "CRUZET4_V2P::All"
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
    '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/058/059/06A2E4C8-976F-DD11-A692-000423D6C8EE.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/058/059/0869DF02-6D6F-DD11-B318-001617DBCF90.root'
    
    )
                            )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2000))

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
