import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersCLUE3DHighL1Seeded_cfi import *
from ..modules.hltHgcalDigis_cfi import *
from ..modules.hltHgcalDigisL1Seeded_cfi import *
from ..modules.hltHgcalLayerClustersEEL1Seeded_cfi import *
from ..modules.hltHgcalLayerClustersHSciL1Seeded_cfi import *
from ..modules.hltHgcalLayerClustersHSiL1Seeded_cfi import *
from ..modules.hltHgcalMergeLayerClustersL1Seeded_cfi import *
from ..modules.hltHGCalRecHitL1Seeded_cfi import *
from ..modules.hltHGCalUncalibRecHitL1Seeded_cfi import *
from ..modules.hltL1TEGammaHGCFilteredCollectionProducer_cfi import *
from ..modules.hltRechitInRegionsHGCAL_cfi import *
from ..modules.hltParticleFlowClusterHGCalFromTICLL1Seeded_cfi import *
from ..modules.hltParticleFlowRecHitHGCL1Seeded_cfi import *
from ..modules.hltParticleFlowSuperClusterHGCalFromTICLL1Seeded_cfi import *
from ..modules.hltTiclLayerTileProducerL1Seeded_cfi import *
from ..modules.hltTiclSeedingL1_cfi import *
from ..modules.hltTiclTrackstersCLUE3DHighL1Seeded_cfi import *
from ..modules.hltTiclTracksterLinksL1Seeded_cfi import *
from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
from RecoHGCal.TICL.ticlEGammaSuperClusterProducer_cfi import ticlEGammaSuperClusterProducer
from Configuration.ProcessModifiers.ticl_superclustering_dnn_cff import ticl_superclustering_dnn
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl

HLTHgcalTiclPFClusteringForEgammaL1SeededSequence = cms.Sequence(hltHgcalDigis+hltL1TEGammaHGCFilteredCollectionProducer+hltHgcalDigisL1Seeded+hltHGCalUncalibRecHitL1Seeded+hltHGCalRecHitL1Seeded+hltParticleFlowRecHitHGCL1Seeded+hltRechitInRegionsHGCAL+hltHgcalLayerClustersEEL1Seeded+hltHgcalLayerClustersHSciL1Seeded+hltHgcalLayerClustersHSiL1Seeded+hltHgcalMergeLayerClustersL1Seeded+hltFilteredLayerClustersCLUE3DHighL1Seeded+hltTiclSeedingL1+hltTiclLayerTileProducerL1Seeded+hltTiclTrackstersCLUE3DHighL1Seeded+hltParticleFlowClusterHGCalFromTICLL1Seeded+hltParticleFlowSuperClusterHGCalFromTICLL1Seeded)


# Enable EGammaSuperClusterProducer at HLT in ticl v5
hltTiclTracksterLinksSuperclusteringDNNL1Seeded = hltTiclTracksterLinksL1Seeded.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringDNN"),
        algo_verbosity=cms.int32(0),
        onnxModelPath = cms.FileInPath("RecoHGCal/TICL/data/superclustering/supercls_v2p1.onnx"),
        nnWorkingPoint=cms.double(0.3),
    ),
    tracksters_collections = [cms.InputTag("hltTiclTrackstersCLUE3DHighL1Seeded")], # to be changed to ticlTrackstersCLUE3DEM once separate CLUE3D iterations are introduced
)

hltTiclTracksterLinksSuperclusteringMustacheL1Seeded = hltTiclTracksterLinksL1Seeded.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringMustache"),
        algo_verbosity=cms.int32(0)
    ),
    tracksters_collections = [cms.InputTag("hltTiclTrackstersCLUE3DHighL1Seeded")], # to be changed to ticlTrackstersCLUE3DEM once separate CLUE3D iterations are introduced
)

hltTiclEGammaSuperClusterProducerL1Seeded = ticlEGammaSuperClusterProducer.clone()

HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_ticlv5_DNN = cms.Sequence(hltHgcalDigis+hltL1TEGammaHGCFilteredCollectionProducer+hltHgcalDigisL1Seeded+hltHGCalUncalibRecHitL1Seeded+hltHGCalRecHitL1Seeded+hltParticleFlowRecHitHGCL1Seeded+hltRechitInRegionsHGCAL+hltHgcalLayerClustersEEL1Seeded+hltHgcalLayerClustersHSciL1Seeded+hltHgcalLayerClustersHSiL1Seeded+hltHgcalMergeLayerClustersL1Seeded+hltFilteredLayerClustersCLUE3DHighL1Seeded+hltTiclSeedingL1+hltTiclLayerTileProducerL1Seeded+hltTiclTrackstersCLUE3DHighL1Seeded+hltTiclTracksterLinksSuperclusteringDNNL1Seeded+hltTiclEGammaSuperClusterProducerL1Seeded)

HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_ticlv5_mustache = cms.Sequence(hltHgcalDigis+hltL1TEGammaHGCFilteredCollectionProducer+hltHgcalDigisL1Seeded+hltHGCalUncalibRecHitL1Seeded+hltHGCalRecHitL1Seeded+hltParticleFlowRecHitHGCL1Seeded+hltRechitInRegionsHGCAL+hltHgcalLayerClustersEEL1Seeded+hltHgcalLayerClustersHSciL1Seeded+hltHgcalLayerClustersHSiL1Seeded+hltHgcalMergeLayerClustersL1Seeded+hltFilteredLayerClustersCLUE3DHighL1Seeded+hltTiclSeedingL1+hltTiclLayerTileProducerL1Seeded+hltTiclTrackstersCLUE3DHighL1Seeded+hltTiclTracksterLinksSuperclusteringMustacheL1Seeded+hltTiclEGammaSuperClusterProducerL1Seeded)

# DNN
ticl_superclustering_dnn.toReplaceWith(HLTHgcalTiclPFClusteringForEgammaL1SeededSequence, HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_ticlv5_DNN)
ticl_superclustering_dnn.toModify(hltTiclEGammaSuperClusterProducerL1Seeded, 
                                  ticlSuperClusters=cms.InputTag("hltTiclTracksterLinksSuperclusteringDNNL1Seeded"),
                                  ticlTrackstersEM=cms.InputTag("hltTiclTrackstersCLUE3DHighL1Seeded"),
                                  layerClusters=cms.InputTag("hltHgcalMergeLayerClustersL1Seeded"))

# Mustache
ticl_superclustering_mustache_ticl.toReplaceWith(HLTHgcalTiclPFClusteringForEgammaL1SeededSequence, HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_ticlv5_mustache)
ticl_superclustering_mustache_ticl.toModify(hltTiclEGammaSuperClusterProducerL1Seeded, 
                                  ticlSuperClusters=cms.InputTag("hltTiclTracksterLinksSuperclusteringMustacheL1Seeded"),
                                  ticlTrackstersEM=cms.InputTag("hltTiclTrackstersCLUE3DHighL1Seeded"),
                                  layerClusters=cms.InputTag("hltHgcalMergeLayerClustersL1Seeded"))