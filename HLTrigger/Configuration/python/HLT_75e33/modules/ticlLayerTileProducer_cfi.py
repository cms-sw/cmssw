import FWCore.ParameterSet.Config as cms

ticlLayerTileProducer = cms.EDProducer("TICLLayerTileProducer",
    detector = cms.string('HGCAL'),
    layer_HFNose_clusters = cms.InputTag("hgcalLayerClustersHFNose"),
    layer_clusters = cms.InputTag("hgcalLayerClusters"),
    mightGet = cms.optional.untracked.vstring
)
