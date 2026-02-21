import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.hltHgcalDigis_cfi import *
from ..modules.hltHgcalLayerClustersEE_cfi import *
from ..modules.hltHgcalLayerClustersHSci_cfi import *
from ..modules.hltHgcalLayerClustersHSi_cfi import *
from ..modules.hltMergeLayerClusters_cfi import *
from ..modules.hltHGCalRecHit_cfi import *
from ..modules.hltHGCalUncalibRecHit_cfi import *
from ..modules.hltParticleFlowClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitHGC_cfi import *
from ..modules.hltTiclLayerTileProducer_cfi import *
from ..modules.hltTiclSeedingGlobal_cfi import *
from ..modules.hltTiclTrackstersCLUE3DHigh_cfi import *
from ..modules.hltHgcalSoARecHitsProducer_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import *
from ..modules.hltHgcalSoALayerClustersProducer_cfi import *
from ..modules.hltHgcalLayerClustersFromSoAProducer_cfi import *
from ..modules.hltTiclTracksterLinks_cfi import *
# Barrel layer clusters
from ..modules.hltBarrelLayerClustersEB_cfi import *
from ..modules.hltBarrelLayerClustersHB_cfi import *



# Use EGammaSuperClusterProducer at HLT in ticl v5
hltTiclTracksterLinksSuperclusteringDNNUnseeded = hltTiclTracksterLinks.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringDNN"),
        algo_verbosity=cms.int32(0),
        onnxModelPath = cms.string("RecoHGCal/TICL/data/superclustering/supercls_v3.onnx"),
        nnWorkingPoint=cms.double(0.57247),
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

from RecoHGCal.TICL.ticlEGammaSuperClusterProducer_cfi import ticlEGammaSuperClusterProducer as _ticlEGammaSuperClusterProducer
hltTiclEGammaSuperClusterProducerUnseeded = _ticlEGammaSuperClusterProducer.clone(
    ticlSuperClusters = "hltTiclTracksterLinksSuperclusteringDNNUnseeded",
    ticlTrackstersEM = "hltTiclTrackstersCLUE3DHigh",
    layerClusters = "hltMergeLayerClusters"
)


_HgcalLocalRecoUnseededSequence = cms.Sequence(hltHgcalDigis+hltHGCalUncalibRecHit+
                                               hltHGCalRecHit+hltParticleFlowRecHitHGC+
                                               hltHgcalLayerClustersEE+
                                               hltHgcalLayerClustersHSci+
                                               hltHgcalLayerClustersHSi+
                                               hltMergeLayerClusters)

_HgcalTICLPatternRecognitionUnseededSequence = cms.Sequence(hltFilteredLayerClustersCLUE3DHigh+
                                                            hltTiclSeedingGlobal+hltTiclLayerTileProducer+
                                                            hltTiclTrackstersCLUE3DHigh)


_SuperclusteringUnseededSequence = cms.Sequence(hltTiclTracksterLinksSuperclusteringDNNUnseeded+ hltTiclEGammaSuperClusterProducerUnseeded)

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
                                  + hltMergeLayerClusters
                     ) 
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
                                            layerClusters=cms.InputTag("hltMergeLayerClusters"),
                                            enableRegression=cms.bool(False)
)

_HgcalLocalRecoUnseededSequence_barrel = _HgcalLocalRecoUnseededSequence.copy()
_HgcalLocalRecoUnseededSequence_barrel += hltBarrelLayerClustersEB
_HgcalLocalRecoUnseededSequence_barrel += hltBarrelLayerClustersHB

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toReplaceWith(_HgcalLocalRecoUnseededSequence, _HgcalLocalRecoUnseededSequence_barrel)

