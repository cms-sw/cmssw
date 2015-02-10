import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMOnlineSimData")

#process.MessageLogger = cms.Service(
#    "MessageLogger",
#    debug_txt = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG') ),
#    destinations = cms.untracked.vstring('debug_txt'),
#    #debugModules = cms.untracked.vstring("SiStripMonitorTrack","SiStripClusterInfo")
#    debugModules = cms.untracked.vstring("SiStripMonitorTrack")
#    )

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
process.p = cms.Path(process.DQMSiStripMonitorTrack_CosmicSim)
process.SiStripMonitorTrack.CCAnalysis_On = True
# Monitoring with Eta Function from SiStripRawDigi
#process.load("DQM.SiStripMonitorTrack.SiStripMonitorTrack_RawStandAlone_cff")
#process.p = cms.Path(process.DQMSiStripMonitorTrack_CosmicRawSim)

# Some changes
process.SiStripMonitorTrack.TrackProducer = 'ctfWithMaterialTracksP5'
process.TrackRefitter.TrajectoryInEvent = True

# To tune Charge Coupling
#---------------------------------------
#SiTrivialInduceChargeOnStrips
# DECOnvolution Mode
# TIB
#process.simSiStripDigis.CouplingCostantDecTIB = cms.vdouble(0.76, 0.12)
# TID
#process.simSiStripDigis.CouplingCostantDecTID = cms.vdouble(0.76, 0.12)
# TOB
#process.simSiStripDigis.CouplingCostantDecTOB = cms.vdouble(0.76, 0.12)
# TEC
#process.simSiStripDigis.CouplingCostantDecTEC = cms.vdouble(0.76, 0.12)
# PEAK Mode
# TIB
#process.simSiStripDigis.CouplingCostantPeakTIB = cms.vdouble(0.94, 0.03)
# TID
#process.simSiStripDigis.CouplingCostantPeakTID = cms.vdouble(0.94, 0.03)
# TOB
#process.simSiStripDigis.CouplingCostantPeakTOB = cms.vdouble(0.94, 0.03)
# TEC
#process.simSiStripDigis.CouplingCostantPeakTEC = cms.vdouble(0.94, 0.03)
#

process.printout = cms.OutputModule("AsciiOutputModule")
process.ep = cms.EndPath(process.printout)

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.source = cms.Source(
    # Underground cosmics CRUZET
    "PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/00F1E965-0B07-DD11-A4E1-0016368E0C2C.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/00F8A7F3-4A06-DD11-B4D1-00E08122B009.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/028783EF-5F06-DD11-A0E8-00163691D6C2.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/02C9E4D4-C60C-DD11-9E29-00E08134A520.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/04130FDF-4706-DD11-83BD-003048359D9C.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/049E40E6-4706-DD11-8FB3-003048770DCC.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/04A414BE-3606-DD11-BC81-003048335516.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/04ADFCBC-4106-DD11-8757-003048770DB6.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/04C6A3CB-C60C-DD11-8373-00E08134420C.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/04DCCF4B-3806-DD11-8954-00E0814295EC.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/0601E98F-4A06-DD11-8D2C-00E081232F37.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/0609F690-4D06-DD11-A8AE-00E081300050.root',
    '/store/mc/2008/4/8/CSA08-CosmicMuonsUndergroundTracker0T-4254/0013/0643CE06-4806-DD11-BA7D-00188B798D8F.root'
   )
    #,firstRun   = cms.untracked.uint32(6),
    #firstEvent = cms.untracked.uint32(15)
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20000)
    )
