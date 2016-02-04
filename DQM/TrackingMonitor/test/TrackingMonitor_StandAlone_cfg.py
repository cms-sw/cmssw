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
# MAGNETIC FIELD & CO
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_3XY_V15::All"
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
process.load("DQM.TrackingMonitor.TrackingMonitor_cfi")

# InputTags for collections 
process.TrackMon.TrackProducer          = cms.InputTag("generalTracks")
process.TrackMon.SeedProducer           = cms.InputTag("newSeedFromTriplets")
process.TrackMon.TCProducer             = cms.InputTag("newTrackCandidateMaker")
process.TrackMon.beamSpot               = cms.InputTag("offlineBeamSpot")

# properties
process.TrackMon.AlgoName               = cms.string('GenTk')
process.TrackMon.OutputFileName 		= cms.string('TrackingMonitor.root')
process.TrackMon.OutputMEsInRootFile 	= cms.bool(True) 
process.TrackMon.FolderName             = cms.string('Track/GlobalParameters')
process.TrackMon.BSFolderName           = cms.string('Track/BeamSpotParameters')
process.TrackMon.MeasurementState 		= cms.string('ImpactPoint')

# which plots to do
process.TrackMon.doTrackerSpecific 		= cms.bool(False)
process.TrackMon.doAllPlots 			= cms.bool(False)
process.TrackMon.doBeamSpotPlots 		= cms.bool(False)
process.TrackMon.doSeedParameterHistos  = cms.bool(False)

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
# redo the tracks
process.preTracking = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits)
process.tracking    = cms.Path(process.ckftracks)

# the main paths
process.trkmon  = cms.Path( process.TrackMon )
process.outP    = cms.OutputModule("AsciiOutputModule")
process.ep      = cms.EndPath(process.outP)

process.seq      = cms.Schedule(
    process.preTracking,
    process.tracking,
    process.trkmon,
    process.ep
)


