import FWCore.ParameterSet.Config as cms


hltTiclEGammaSuperClusterProducerL1Seeded = cms.EDProducer('EGammaSuperclusterProducer',
    ticlSuperClusters = cms.InputTag('hltTiclTracksterLinksSuperclusteringDNNL1Seeded'),
    ticlTrackstersEM = cms.InputTag('hltTiclTrackstersCLUE3DHighL1Seeded'),
    layerClusters = cms.InputTag('hltMergeLayerClustersL1Seeded'),
    superclusterEtThreshold = cms.float(4),
    enableRegression = cms.bool(True),
    regressionModelPath = cms.FileInPath('RecoHGCal/TICL/data/superclustering/regression_v1.onnx'),
    mightGet = cms.optional.untracked.vstring
)

from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl

ticl_superclustering_mustache_ticl.toModify(hltTiclEGammaSuperClusterProducerL1Seeded, 
                                            ticlSuperClusters=cms.InputTag("hltTiclTracksterLinksSuperclusteringMustacheL1Seeded"),
                                            enableRegression=cms.bool(False)
)