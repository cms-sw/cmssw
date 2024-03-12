import FWCore.ParameterSet.Config as cms

ticlLayerTileProducerL1Seeded = cms.EDProducer("TICLLayerTileProducer",
    detector = cms.string('HGCAL'),
    layer_HFNose_clusters = cms.InputTag("hgcalLayerClustersHFNose"),
    layer_clusters = cms.InputTag("hgcalMergeLayerClustersL1Seeded"),
    mightGet = cms.optional.untracked.vstring
)
# foo bar baz
# zd5QQ6Dy5ehyy
# K7T0K1RJ2NDGv
