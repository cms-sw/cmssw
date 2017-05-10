import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import *

siStripClusters = cms.EDProducer("SiStripClusterizerTagMCmerged",
                               Clusterizer = DefaultClusterizer,
                               DigiProducersList = cms.VInputTag(
    cms.InputTag('siStripDigis','ZeroSuppressed'),
    cms.InputTag('siStripZeroSuppression','VirginRaw'),
    cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
    cms.InputTag('siStripZeroSuppression','ScopeMode')),
                               )
#  For TrackerHitAssociator
siStripClusters.Clusterizer.ROUList = cms.vstring('g4SimHitsTrackerHitsTIBLowTof',
                      'g4SimHitsTrackerHitsTIBHighTof',
                      'g4SimHitsTrackerHitsTIDLowTof',
                      'g4SimHitsTrackerHitsTIDHighTof',
                      'g4SimHitsTrackerHitsTOBLowTof',
                      'g4SimHitsTrackerHitsTOBHighTof',
                      'g4SimHitsTrackerHitsTECLowTof',
                      'g4SimHitsTrackerHitsTECHighTof')
siStripClusters.Clusterizer.associateRecoTracks = cms.bool(True)  # True to save some time if no PU
siStripClusters.Clusterizer.associatePixel = cms.bool(False)
siStripClusters.Clusterizer.associateStrip = cms.bool(True)
