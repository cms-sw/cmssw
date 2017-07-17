import FWCore.ParameterSet.Config as cms

siStripClusters = cms.EDProducer("ClusterRefinerTagMCmerged",
  UntaggedClusterProducer = cms.InputTag('siStripClustersUntagged'),
  ClusterRefiner = cms.PSet(
#   For TrackerHitAssociator
    ROUList = cms.vstring('g4SimHitsTrackerHitsTIBLowTof',
                          'g4SimHitsTrackerHitsTIBHighTof',
                          'g4SimHitsTrackerHitsTIDLowTof',
                          'g4SimHitsTrackerHitsTIDHighTof',
                          'g4SimHitsTrackerHitsTOBLowTof',
                          'g4SimHitsTrackerHitsTOBHighTof',
                          'g4SimHitsTrackerHitsTECLowTof',
                          'g4SimHitsTrackerHitsTECHighTof'),
    associateRecoTracks = cms.bool(True),  # True to save some time if no PU
    associatePixel = cms.bool(False),
    associateStrip = cms.bool(True)
  )
)
