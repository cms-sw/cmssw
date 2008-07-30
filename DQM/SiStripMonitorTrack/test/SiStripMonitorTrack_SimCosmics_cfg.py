import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMOnlineSimData")

process.MessageLogger = cms.Service(
    "MessageLogger",
    debug_txt = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG') ),
    destinations = cms.untracked.vstring('debug_txt'),
    #debugModules = cms.untracked.vstring("SiStripMonitorTrack","SiStripClusterInfo")
    debugModules = cms.untracked.vstring("SiStripMonitorTrack")
    )

#-------------------------------------------------
## MAGNETIC FIELD
#-------------------------------------------------
# Magnetic field: force mag field to be 0.0 tesla
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

#-------------------------------------------------
## GEOMETRY
#-------------------------------------------------

# CMS Geometry
process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
## CALIBRATION
#-------------------------------------------------
process.load("CalibTracker.Configuration.Tracker_FakeConditions_cff")

#--------------------------
#### DQM
#--------------------------
# DQM services
process.load("DQMServices.Core.DQM_cfg")

#-------------------------------------
#### SiStripMonitorTrack + Scheduling
#-------------------------------------
# Standard Monitoring
#process.load("DQM.SiStripMonitorTrack.SiStripMonitorTrack_StandAlone_cff")
#process.p = cms.Path(process.DQMSiStripMonitorTrack_CosmicSim)
# Monitoring with Eta Function from SiStripRawDigi
process.load("DQM.SiStripMonitorTrack.SiStripMonitorTrack_RawStandAlone_cff")
process.p = cms.Path(process.DQMSiStripMonitorTrack_CosmicRawSim)

# Some changes
process.SiStripMonitorTrack.TrackProducer = 'ctfWithMaterialTracksP5'
process.TrackRefitter.TrajectoryInEvent = True

process.printout = cms.OutputModule("AsciiOutputModule")
process.ep = cms.EndPath(process.printout)

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.source = cms.Source(
    # Underground cosmics CRUZET
    "PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/00F1E965-0B07-DD11-A4E1-0016368E0C2C.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/00F8A7F3-4A06-DD11-B4D1-00E08122B009.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/028783EF-5F06-DD11-A0E8-00163691D6C2.root'
    )
    #,firstRun   = cms.untracked.uint32(6),
    #firstEvent = cms.untracked.uint32(15)
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )
