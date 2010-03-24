import FWCore.ParameterSet.Config as cms

process = cms.Process("clusterAnalysis")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#    '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/BSCNOBEAMHALO-Dec19thSkim_336p3_v1/0008/1630A134-E0F0-DE11-8A34-001D0967DC0B.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/BSCNOBEAMHALO-Dec19thSkim_336p3_v1/0006/EAC5876C-E6EE-DE11-9FA5-0024E87687CB.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/BSCNOBEAMHALO-Dec19thSkim_336p3_v1/0006/10F36BBF-D9EE-DE11-90B1-0024E87683B7.root'
#     '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_336p3_v2/0102/B4237151-29ED-DE11-81ED-0015178C1804.root',
#     '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_336p3_v2/0102/A674A9FE-19ED-DE11-8C16-00151785FF78.root',
#     '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_336p3_v2/0102/A622C6B7-18ED-DE11-AB28-0024E8768C23.root',
#     '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_336p3_v2/0102/94AB71E7-1BED-DE11-AE0C-001D0967D314.root',
#     '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_336p3_v2/0102/745BD05C-21ED-DE11-A20D-001D0967CF77.root',
#     '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_336p3_v2/0102/66001B96-10ED-DE11-9728-0024E8767DA0.root',
#     '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_336p3_v2/0102/223763C6-18ED-DE11-89A4-001D0967D49F.root',
     '/store/data/BeamCommissioning09/MinimumBias/RECO/Feb9ReReco_v2/0025/E09435BA-E515-DF11-A78F-003048679180.root',
     '/store/data/BeamCommissioning09/MinimumBias/RECO/Feb9ReReco_v2/0025/E0599DBB-E415-DF11-A592-00304867915A.root',
     '/store/data/BeamCommissioning09/MinimumBias/RECO/Feb9ReReco_v2/0025/B67525A1-DF15-DF11-8FF1-0026189437EC.root',
     '/store/data/BeamCommissioning09/MinimumBias/RECO/Feb9ReReco_v2/0025/B0CD2A8F-E015-DF11-AF38-0026189438B0.root',
     '/store/data/BeamCommissioning09/MinimumBias/RECO/Feb9ReReco_v2/0025/76E9248C-DF15-DF11-8CE8-00261894387A.root',
     '/store/data/BeamCommissioning09/MinimumBias/RECO/Feb9ReReco_v2/0025/68AFEEF1-ED15-DF11-A38C-00261894398D.root',
     '/store/data/BeamCommissioning09/MinimumBias/RECO/Feb9ReReco_v2/0025/0A730102-E815-DF11-9E08-003048D42D92.root',
     '/store/data/BeamCommissioning09/MinimumBias/RECO/Feb9ReReco_v2/0025/00AB64B9-E515-DF11-9032-003048678B5E.root'
)
)

# Conditions (Global Tag is used here):
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR09_R_34X_V5::All'
#process.GlobalTag.globaltag = 'GR09_R_35X_V3::All'

#Geometry and field
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("TrackingTools.RecoGeometry.RecoGeometries_cff")

#tracker refitting -> trajectory
process.load('RecoTracker.TrackProducer.TrackRefitters_cff')
process.ttrhbwr.ComputeCoarseLocalPositionFromDisk = True
process.generalTracks = process.TrackRefitter.clone(
   src = cms.InputTag("generalTracks")
)
process.ctfPixelLess = process.TrackRefitter.clone(
   src = cms.InputTag("ctfPixelLess")
)
process.refit = cms.Sequence(process.generalTracks*process.ctfPixelLess*process.doAlldEdXEstimators)
## re_fitting
#process.load('Configuration/StandardSequences/Reconstruction_cff')
#process.refit = cms.Sequence(
#    process.siPixelRecHits * 
#    process.siStripMatchedRecHits *
#    process.ckftracks *
#    process.ctfTracksPixelLess
#)

#analysis
process.analysis = cms.EDAnalyzer('TrackerDpgAnalysis',
   ClustersLabel = cms.InputTag("siStripClusters"),
   PixelClustersLabel = cms.InputTag("siPixelClusters"),
   TracksLabel   = cms.VInputTag( cms.InputTag("generalTracks"), cms.InputTag("ctfPixelLess") ),
   vertexLabel   = cms.InputTag('offlinePrimaryVertices'),
   pixelVertexLabel = cms.InputTag('pixelVertices'),
   beamSpotLabel = cms.InputTag('offlineBeamSpot'),
   DeDx1Label    = cms.InputTag('dedxHarmonic2'),
   DeDx2Label    = cms.InputTag('dedxTruncated40'),
   DeDx3Label    = cms.InputTag('dedxMedian'),
   L1Label       = cms.InputTag('gtDigis'),
   HLTLabel      = cms.InputTag("TriggerResults"),
   InitalCounter = cms.uint32(1),
   keepOntrackClusters  = cms.untracked.bool(False),
   keepOfftrackClusters = cms.untracked.bool(False),
   keepPixelClusters    = cms.untracked.bool(False),
   keepPixelVertices    = cms.untracked.bool(False),
   keepMissingHits      = cms.untracked.bool(False),
   keepTracks           = cms.untracked.bool(True),
   keepVertices         = cms.untracked.bool(True),
   keepEvents           = cms.untracked.bool(True),
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('trackerDPG_124120_small.root')
)

process.skimming = cms.EDFilter("PhysDecl",
  applyfilter = cms.untracked.bool(True)
)

process.p = cms.Path(process.skimming*process.refit*process.analysis)
#process.dump = cms.EDAnalyzer("EventContentAnalyzer")
#process.p = cms.Path(process.refit*process.dump)
