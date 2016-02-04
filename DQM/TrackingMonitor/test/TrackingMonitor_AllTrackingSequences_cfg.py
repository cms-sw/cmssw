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
    fileNames = cms.untracked.vstring('/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0005/A0AFB73F-38D7-DE11-82F3-0026189438B4.root')
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))


#-------------------------------------------------
# MAGNETIC FIELD & RECO
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_3XY_V15::All"
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
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
process.tracking    = cms.Path(process.ckftracks_plus_pixelless) 
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


