import FWCore.ParameterSet.Config as cms


hltTiclEGammaSuperClusterProducerUnseeded = cms.EDProducer('EGammaSuperclusterProducer',
    ticlSuperClusters = cms.InputTag('hltTiclTracksterLinksSuperclusteringDNNUnseeded'),
    ticlTrackstersEM = cms.InputTag('hltTiclTrackstersCLUE3DHigh'),
    layerClusters = cms.InputTag('hltMergeLayerClusters'),
    superclusterEtThreshold = cms.double(4),
    enableRegression = cms.bool(True),
    regressionModelPath = cms.FileInPath('RecoHGCal/TICL/data/superclustering/regression_v1.onnx'),
    mightGet = cms.optional.untracked.vstring
)

from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl

ticl_superclustering_mustache_ticl.toModify(hltTiclEGammaSuperClusterProducerUnseeded, 
                                            ticlSuperClusters=cms.InputTag("hltTiclTracksterLinksSuperclusteringMustacheUnseeded"),
                                            enableRegression=cms.bool(False)
)