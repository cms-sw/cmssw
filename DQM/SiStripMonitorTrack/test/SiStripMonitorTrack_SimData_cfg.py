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
process.load("Configuration.StandardSequences.MagneticField_0T_cff")
process.prefer("VolumeBasedMagneticFieldESProducer")

#-------------------------------------------------
## GEOMETRY
#-------------------------------------------------

# CMS Geometry
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

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
process.load("DQM.SiStripMonitorTrack.SiStripMonitorTrack_StandAlone_cff")
process.p = cms.Path(process.DQMSiStripMonitorTrack_Sim)
# Monitoring with Eta Function from SiStripRawDigi
#process.load("DQM.SiStripMonitorTrack.SiStripMonitorTrack_RawStandAlone_cff")
#process.p = cms.Path(process.DQMSiStripMonitorTrack_RawSim)
process.TrackRefitter.TrajectoryInEvent = True

process.printout = cms.OutputModule("AsciiOutputModule")
process.ep = cms.EndPath(process.printout)

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/6/RelVal-RelValSingleMuPt1-1212531852-IDEAL_V1-2nd-02/0000/74A33D36-E933-DD11-BC9E-001617E30F56.root')
    #firstRun   = cms.untracked.uint32(6)
    #firstEvent = cms.untracked.uint32(15)
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )
