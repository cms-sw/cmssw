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
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# TkDetMap for TkHistoMap
#-------------------------------------------------
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#-process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_30X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRAFT_ALL_V8::All"
process.GlobalTag.globaltag = "CRAFT_30X::All"
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
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/46FF6C90-71C1-DD11-A71F-0019B9E4FFE1.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/8A9B693E-46C1-DD11-88D1-001D0967D558.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0000/9ACFD2EC-67C1-DD11-8CC2-001D0967CFCC.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/1EF817E4-CEC1-DD11-82F4-001D0967CFCC.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/3CB465BE-03C2-DD11-949C-001D0967D48B.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/5A00515D-B5C1-DD11-B6CE-001D0967D616.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0001/EEF3EDF1-D8C1-DD11-BBF7-0019B9E4FC1C.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0002/B0A82B97-1FC2-DD11-8734-001D0968F684.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0003/8494C44F-63C2-DD11-87F6-0019B9E4FF87.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V4_ReReco-v1/0004/1A306101-93C2-DD11-A126-001D0967D512.root'

    

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
