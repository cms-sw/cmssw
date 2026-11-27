import FWCore.ParameterSet.Config as cms

hltTiclLayerTileProducerL1Seeded = cms.EDProducer("TICLLayerTileProducer",
    ticlGeom = cms.ESInputTag('', 'hlt'),
    detector = cms.string('HGCAL'),
    layer_HFNose_clusters = cms.InputTag("hgcalLayerClustersHFNose"),
    layer_clusters = cms.InputTag("hltMergeLayerClustersL1Seeded"),
    mightGet = cms.optional.untracked.vstring
)
