import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.hltHgcalDigis_cfi import *
from ..modules.hltHgcalLayerClustersEE_cfi import *
from ..modules.hltHgcalLayerClustersHSci_cfi import *
from ..modules.hltHgcalLayerClustersHSi_cfi import *
from ..modules.hltHgcalMergeLayerClusters_cfi import *
from ..modules.hltHGCalRecHit_cfi import *
from ..modules.hltHGCalUncalibRecHit_cfi import *
from ..modules.hltParticleFlowClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitHGC_cfi import *
from ..modules.hltParticleFlowSuperClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.hltTiclLayerTileProducer_cfi import *
from ..modules.hltTiclSeedingGlobal_cfi import *
from ..modules.hltTiclTrackstersCLUE3DHigh_cfi import *
from ..modules.hltHgcalSoARecHitsProducer_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import *
from ..modules.hltHgcalSoALayerClustersProducer_cfi import *
from ..modules.hltHgcalLayerClustersFromSoAProducer_cfi import *
from ..modules.hltTiclTracksterLinksUnseeded_cfi import *
from RecoHGCal.TICL.ticlEGammaSuperClusterProducer_cfi import ticlEGammaSuperClusterProducer
from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
from Configuration.ProcessModifiers.ticl_superclustering_dnn_cff import ticl_superclustering_dnn
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl

HLTHgcalTiclPFClusteringForEgammaUnseededSequence = cms.Sequence(hltHgcalDigis+hltHGCalUncalibRecHit+hltHGCalRecHit+hltParticleFlowRecHitHGC+hltHgcalLayerClustersEE+hltHgcalLayerClustersHSci+hltHgcalLayerClustersHSi+hltHgcalMergeLayerClusters+hltFilteredLayerClustersCLUE3DHigh+hltTiclSeedingGlobal+hltTiclLayerTileProducer+hltTiclTrackstersCLUE3DHigh+hltParticleFlowClusterHGCalFromTICLUnseeded+hltParticleFlowSuperClusterHGCalFromTICLUnseeded)

#_HLTHgcalTiclPFClusteringForEgammaUnseededSequence_heterogeneous = cms.Sequence(hltHgcalDigis+hltHGCalUncalibRecHit+hltHGCalRecHit+hltParticleFlowRecHitHGC+hltHgcalSoARecHitsProducer+hltHgcalSoARecHitsLayerClustersProducer+hltHgcalSoALayerClustersProducer+hltHgCalLayerClustersFromSoAProducer+hltHgcalLayerClustersHSci+hltHgcalLayerClustersHSi+hltHgcalMergeLayerClusters+hltFilteredLayerClustersCLUE3DHigh+hltTiclSeedingGlobal+hltTiclLayerTileProducer+hltTiclTrackstersCLUE3DHigh+hltParticleFlowClusterHGCalFromTICLUnseeded+hltParticleFlowSuperClusterHGCalFromTICLUnseeded)

#from Configuration.ProcessModifiers.alpaka_cff import alpaka
#alpaka.toReplaceWith(HLTHgcalTiclPFClusteringForEgammaUnseededSequence, _HLTHgcalTiclPFClusteringForEgammaUnseededSequence_heterogeneous)
#alpaka.toModify(hltHgcalMergeLayerClusters,
#        layerClustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer"),
#        time_layerclustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer", "timeLayerCluster"))


# Enable EGammaSuperClusterProducer at HLT in ticl v5

hltTiclTracksterLinksSuperclusteringDNNUnseeded = hltTiclTracksterLinksUnseeded.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringDNN"),
        algo_verbosity=cms.int32(0),
        onnxModelPath = cms.FileInPath("RecoHGCal/TICL/data/superclustering/supercls_v2p1.onnx"),
        nnWorkingPoint=cms.double(0.3),
    ),
    tracksters_collections = [cms.InputTag("hltTiclTrackstersCLUE3DHigh")], # to be changed to ticlTrackstersCLUE3DEM once separate CLUE3D iterations are introduced
)

hltTiclTracksterLinksSuperclusteringMustacheUnseeded = hltTiclTracksterLinksUnseeded.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringMustache"),
        algo_verbosity=cms.int32(0)
    ),
    tracksters_collections = [cms.InputTag("hltTiclTrackstersCLUE3DHigh")], # to be changed to ticlTrackstersCLUE3DEM once separate CLUE3D iterations are introduced
)

hltTiclEGammaSuperClusterProducerUnseeded = ticlEGammaSuperClusterProducer.clone()

HLTHgcalTiclPFClusteringForEgammaUnseededSequence_ticlv5_DNN = cms.Sequence(hltHgcalDigis+hltHGCalUncalibRecHit+hltHGCalRecHit+hltParticleFlowRecHitHGC+hltHgcalLayerClustersEE+hltHgcalLayerClustersHSci+hltHgcalLayerClustersHSi+hltHgcalMergeLayerClusters+hltFilteredLayerClustersCLUE3DHigh+hltTiclSeedingGlobal+hltTiclLayerTileProducer+hltTiclTrackstersCLUE3DHigh+hltTiclTracksterLinksSuperclusteringDNNUnseeded+hltTiclEGammaSuperClusterProducerUnseeded)

HLTHgcalTiclPFClusteringForEgammaUnseededSequence_ticlv5_mustache = cms.Sequence(hltHgcalDigis+hltHGCalUncalibRecHit+hltHGCalRecHit+hltParticleFlowRecHitHGC+hltHgcalLayerClustersEE+hltHgcalLayerClustersHSci+hltHgcalLayerClustersHSi+hltHgcalMergeLayerClusters+hltFilteredLayerClustersCLUE3DHigh+hltTiclSeedingGlobal+hltTiclLayerTileProducer+hltTiclTrackstersCLUE3DHigh+hltTiclTracksterLinksSuperclusteringMustacheUnseeded+hltTiclEGammaSuperClusterProducerUnseeded)

# DNN
ticl_superclustering_dnn.toReplaceWith(HLTHgcalTiclPFClusteringForEgammaUnseededSequence, HLTHgcalTiclPFClusteringForEgammaUnseededSequence_ticlv5_DNN)
ticl_superclustering_dnn.toModify(hltTiclEGammaSuperClusterProducerUnseeded, 
                                  ticlSuperClusters=cms.InputTag("hltTiclTracksterLinksSuperclusteringDNNUnseeded"),
                                  ticlTrackstersEM=cms.InputTag("hltTiclTrackstersCLUE3DHigh"),
                                  layerClusters=cms.InputTag("hltHgcalMergeLayerClusters"))

# Mustache
ticl_superclustering_mustache_ticl.toReplaceWith(HLTHgcalTiclPFClusteringForEgammaUnseededSequence, HLTHgcalTiclPFClusteringForEgammaUnseededSequence_ticlv5_mustache)
ticl_superclustering_mustache_ticl.toModify(hltTiclEGammaSuperClusterProducerUnseeded, 
                                  ticlSuperClusters=cms.InputTag("hltTiclTracksterLinksSuperclusteringMustacheUnseeded"),
                                  ticlTrackstersEM=cms.InputTag("hltTiclTrackstersCLUE3DHigh"),
                                  layerClusters=cms.InputTag("hltHgcalMergeLayerClusters"))