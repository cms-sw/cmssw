import FWCore.ParameterSet.Config as cms

ticlLayerTileProducer = cms.EDProducer("TICLLayerTileProducer",
    detector = cms.string('HGCAL'),
    layer_HFNose_clusters = cms.InputTag("hgcalLayerClustersHFNose"),
    layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
    mightGet = cms.optional.untracked.vstring
)
# foo bar baz
# 9KFR18wqaAV0l
# PpRJVM7ZBHZ7X
