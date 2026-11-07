import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.hltHgcalDigis_cfi import *
from ..modules.hltMergeLayerClusters_cfi import *
from ..sequences.HLTHgcalLayerClustersSequence_cfi import *
from ..modules.hltHGCalRecHit_cfi import *
from ..modules.hltHGCalUncalibRecHit_cfi import *
from ..modules.hltParticleFlowClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitHGC_cfi import *
from ..modules.hltTiclLayerTileProducer_cfi import *
from ..modules.hltTiclSeedingGlobal_cfi import *
from ..modules.hltTiclTrackstersCLUE3DHigh_cfi import *
from ..modules.hltTiclTracksterLinks_cfi import *
# Barrel layer clusters
from ..modules.hltBarrelLayerClustersEB_cfi import *
from ..modules.hltBarrelLayerClustersHB_cfi import *

from ..modules.hltTiclEGammaSuperClusterProducerUnseeded_cfi import hltTiclEGammaSuperClusterProducerUnseeded
from ..modules.hltTiclTracksterLinksSuperclusteringMustacheUnseeded_cfi import hltTiclTracksterLinksSuperclusteringMustacheUnseeded
from ..modules.hltTiclTracksterLinksSuperclusteringDNNUnseeded_cfi import hltTiclTracksterLinksSuperclusteringDNNUnseeded

# Barrel tracksters
from ..modules.hltFilteredLayerClustersCLUE3DBarrel_cfi import *
from ..modules.hltTiclLayerTileBarrelProducer_cfi import *
from ..modules.hltTiclTrackstersCLUE3DBarrel_cfi import *

_HgcalLocalRecoUnseededSequence = cms.Sequence(hltHgcalDigis+hltHGCalUncalibRecHit+
                                               hltHGCalRecHit+hltParticleFlowRecHitHGC+
                                               HLTHgcalLayerClustersSequence+
                                               hltMergeLayerClusters)

_HgcalTICLPatternRecognitionUnseededSequence = cms.Sequence(hltFilteredLayerClustersCLUE3DHigh+
                                                            hltTiclSeedingGlobal+hltTiclLayerTileProducer+
                                                            hltTiclTrackstersCLUE3DHigh)


_SuperclusteringUnseededSequence = cms.Sequence(hltTiclTracksterLinksSuperclusteringDNNUnseeded+ hltTiclEGammaSuperClusterProducerUnseeded)

HLTHgcalTiclPFClusteringForEgammaUnseededSequence = cms.Sequence(_HgcalLocalRecoUnseededSequence + _HgcalTICLPatternRecognitionUnseededSequence + _SuperclusteringUnseededSequence)





# Ticl mustache
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl
ticl_superclustering_mustache_ticl.toReplaceWith(_SuperclusteringUnseededSequence, 
                                                 cms.Sequence(
                                                              hltTiclTracksterLinksSuperclusteringMustacheUnseeded
                                                              + hltTiclEGammaSuperClusterProducerUnseeded
                                                 )
)


_HgcalLocalRecoUnseededSequence_barrel = _HgcalLocalRecoUnseededSequence.copy()

_HgcalLocalRecoUnseededSequence_barrel = cms.Sequence(hltHgcalDigis+hltHGCalUncalibRecHit+
                                                      hltHGCalRecHit+hltParticleFlowRecHitHGC+
                                                      HLTHgcalLayerClustersSequence+
                                                      hltBarrelLayerClustersEB+
                                                      hltBarrelLayerClustersHB+
                                                      hltMergeLayerClusters)

_HgcalTICLPatternRecognitionUnseededSequence_barrel = cms.Sequence(hltFilteredLayerClustersCLUE3DHigh+
                                                                   hltFilteredLayerClustersCLUE3DBarrel+
                                                                   hltTiclSeedingGlobal+
                                                                   hltTiclLayerTileProducer+
                                                                   hltTiclLayerTileBarrelProducer+
                                                                   hltTiclTrackstersCLUE3DHigh+
                                                                   hltTiclTrackstersCLUE3DBarrel)


from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toReplaceWith(_HgcalLocalRecoUnseededSequence, _HgcalLocalRecoUnseededSequence_barrel)
ticl_barrel.toReplaceWith(_HgcalTICLPatternRecognitionUnseededSequence, _HgcalTICLPatternRecognitionUnseededSequence_barrel)
