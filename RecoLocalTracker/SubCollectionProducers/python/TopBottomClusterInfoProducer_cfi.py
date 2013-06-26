import FWCore.ParameterSet.Config as cms

topBottomClusterInfoProducer = cms.EDProducer("TopBottomClusterInfoProducer",
    stripClustersOld = cms.InputTag("siStripClusters"),
    pixelClustersOld = cms.InputTag("siPixelClusters"),
    stripClustersNew = cms.InputTag("siStripClustersTop"),
    pixelClustersNew = cms.InputTag("siPixelClustersTop"),
    stripMonoHitsOld = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    stripStereoHitsOld = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    pixelHitsOld = cms.InputTag("siPixelRecHits"),
    stripMonoHitsNew = cms.InputTag("siStripMatchedRecHitsTop","rphiRecHit"),
    stripStereoHitsNew = cms.InputTag("siStripMatchedRecHitsTop","stereoRecHit"),
    pixelHitsNew = cms.InputTag("siPixelRecHitsTop")
)
