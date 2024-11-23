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
from ..modules.hltTiclTracksterLinks_cfi import *

_HgcalLocalRecoUnseededSequence = cms.Sequence(hltHgcalDigis+hltHGCalUncalibRecHit+hltHGCalRecHit+hltParticleFlowRecHitHGC+hltHgcalLayerClustersEE+hltHgcalLayerClustersHSci+hltHgcalLayerClustersHSi+hltHgcalMergeLayerClusters)
_HgcalTICLPatternRecognitionUnseededSequence = cms.Sequence(hltFilteredLayerClustersCLUE3DHigh+hltTiclSeedingGlobal+hltTiclLayerTileProducer+hltTiclTrackstersCLUE3DHigh)
_SuperclusteringUnseededSequence = cms.Sequence(hltParticleFlowClusterHGCalFromTICLUnseeded+hltParticleFlowSuperClusterHGCalFromTICLUnseeded)

# The baseline sequence
HLTHgcalTiclPFClusteringForEgammaUnseededSequence = cms.Sequence(_HgcalLocalRecoUnseededSequence + _HgcalTICLPatternRecognitionUnseededSequence + _SuperclusteringUnseededSequence)

# Alpaka
from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toReplaceWith(_HgcalLocalRecoUnseededSequence, 
                     cms.Sequence(
                                  hltHgcalDigis
                                  + hltHGCalUncalibRecHit
                                  + hltHGCalRecHit+hltParticleFlowRecHitHGC
                                  + hltHgcalSoARecHitsProducer
                                  + hltHgcalSoARecHitsLayerClustersProducer
                                  + hltHgcalSoALayerClustersProducer
                                  + hltHgCalLayerClustersFromSoAProducer
                                  + hltHgcalLayerClustersHSci
                                  + hltHgcalLayerClustersHSi
                                  + hltHgcalMergeLayerClusters
                     ) 
)
alpaka.toModify(hltHgcalMergeLayerClusters,
                layerClustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer"),
                time_layerclustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer", "timeLayerCluster")
)

# Use EGammaSuperClusterProducer at HLT in ticl v5
hltTiclTracksterLinksSuperclusteringDNNUnseeded = hltTiclTracksterLinks.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringDNN"),
        algo_verbosity=cms.int32(0),
        onnxModelPath = cms.FileInPath("RecoHGCal/TICL/data/superclustering/supercls_v2p1.onnx"),
        nnWorkingPoint=cms.double(0.3),
    ),
    tracksters_collections = [cms.InputTag("hltTiclTrackstersCLUE3DHigh")], # to be changed to ticlTrackstersCLUE3DEM once separate CLUE3D iterations are introduced
)

hltTiclTracksterLinksSuperclusteringMustacheUnseeded = hltTiclTracksterLinks.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringMustache"),
        algo_verbosity=cms.int32(0)
    ),
    tracksters_collections = [cms.InputTag("hltTiclTrackstersCLUE3DHigh")], # to be changed to ticlTrackstersCLUE3DEM once separate CLUE3D iterations are introduced
)

from RecoHGCal.TICL.ticlEGammaSuperClusterProducer_cfi import ticlEGammaSuperClusterProducer
hltTiclEGammaSuperClusterProducerUnseeded = ticlEGammaSuperClusterProducer.clone()

from Configuration.ProcessModifiers.ticl_superclustering_dnn_cff import ticl_superclustering_dnn
ticl_superclustering_dnn.toReplaceWith(_SuperclusteringUnseededSequence, 
                                       cms.Sequence(
                                                    hltTiclTracksterLinksSuperclusteringDNNUnseeded
                                                    + hltTiclEGammaSuperClusterProducerUnseeded
                                       )
)
ticl_superclustering_dnn.toModify(hltTiclEGammaSuperClusterProducerUnseeded,  
                                  ticlSuperClusters=cms.InputTag("hltTiclTracksterLinksSuperclusteringDNNUnseeded"),
                                  ticlTrackstersEM=cms.InputTag("hltTiclTrackstersCLUE3DHigh"),
                                  layerClusters=cms.InputTag("hltHgcalMergeLayerClusters")
)

# Ticl mustache
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl
ticl_superclustering_mustache_ticl.toReplaceWith(_SuperclusteringUnseededSequence, 
                                                 cms.Sequence(
                                                              hltTiclTracksterLinksSuperclusteringMustacheUnseeded
                                                              + hltTiclEGammaSuperClusterProducerUnseeded
                                                 )
)
ticl_superclustering_mustache_ticl.toModify(hltTiclEGammaSuperClusterProducerUnseeded, 
                                            ticlSuperClusters=cms.InputTag("hltTiclTracksterLinksSuperclusteringMustacheUnseeded"),
                                            ticlTrackstersEM=cms.InputTag("hltTiclTrackstersCLUE3DHigh"),
                                            layerClusters=cms.InputTag("hltHgcalMergeLayerClusters"),
                                            enableRegression=cms.bool(False)
)
