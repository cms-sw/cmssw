import FWCore.ParameterSet.Config as cms

hltFilteredLayerClustersSoAProducer = cms.EDProducer('FilteredLayerClustersSoAProducer@alpaka',
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    LayerClustersSoA = cms.InputTag('hltHgcalSoALayerClustersProducer'),
    min_cluster_size = cms.int32(0),
    max_cluster_size = cms.int32(9999),
    min_layerId = cms.int32(0),
    max_layerId = cms.int32(9999),
    iteration_label = cms.string('iterationLabelGoesHere')
)