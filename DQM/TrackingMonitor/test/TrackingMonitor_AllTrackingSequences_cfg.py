import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# DQMServices
#-------------------------------------------------
process = cms.Process("DQMTrackMon")

#-------------------------------------------------
# message logger
#-------------------------------------------------
process.MessageLogger = cms.Service("MessageLogger",
     default = cms.untracked.PSet( limit = cms.untracked.int32(10) )
)

#-------------------------------------------------
# Source  
#-------------------------------------------------
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/10CF910B-E057-E011-9B13-000423D9A2AE.root')
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))


#-------------------------------------------------
# MAGNETIC FIELD & RECO
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR_R_43_V3::All"
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
process.load("DQM.TrackingMonitor.TrackingMonitorAllTrackingSequences_cff")

#-------------------------------------------------

# DQM Store 
#-------------------------------------------------
process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

#-------------------------------------------------
# Paths 
#-------------------------------------------------

# redo tracking (to get all the track builing and interative steps)
process.preTracking = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits)
process.tracking    = cms.Path(process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks_plus_pixelless) 
#process.tracking    = cms.Path(process.ckftracks)

# tracking monitor paths
process.trackmon    = cms.Path(process.trkmon)

# end path
process.outP    = cms.OutputModule("AsciiOutputModule")
process.ep      = cms.EndPath(process.outP)

process.trkmon = cms.Schedule(
      process.preTracking
    , process.tracking
    , process.trackmon 
    , process.ep
)


