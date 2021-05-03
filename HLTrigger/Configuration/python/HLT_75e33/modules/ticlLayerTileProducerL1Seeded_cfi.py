import FWCore.ParameterSet.Config as cms

ticlLayerTileProducerL1Seeded = cms.EDProducer("TICLLayerTileProducer",
    detector = cms.string('HGCAL'),
    layer_HFNose_clusters = cms.InputTag("hgcalLayerClustersHFNose"),
    layer_clusters = cms.InputTag("hgcalLayerClustersL1Seeded"),
    mightGet = cms.optional.untracked.vstring
)
