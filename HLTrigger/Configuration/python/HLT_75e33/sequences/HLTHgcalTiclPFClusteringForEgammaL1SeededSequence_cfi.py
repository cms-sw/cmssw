import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersCLUE3DHighL1Seeded_cfi import *
from ..modules.hltHgcalDigis_cfi import *
from ..modules.hltHgcalDigisL1Seeded_cfi import *
from ..modules.hltHgcalLayerClustersEEL1Seeded_cfi import *
from ..modules.hltHgcalLayerClustersHSciL1Seeded_cfi import *
from ..modules.hltHgcalLayerClustersHSiL1Seeded_cfi import *
from ..modules.hltMergeLayerClustersL1Seeded_cfi import *
from ..modules.hltHGCalRecHitL1Seeded_cfi import *
from ..modules.hltHGCalUncalibRecHitL1Seeded_cfi import *
from ..modules.hltL1TEGammaHGCFilteredCollectionProducer_cfi import *
from ..modules.hltRechitInRegionsHGCAL_cfi import *
from ..modules.hltParticleFlowClusterHGCalFromTICLL1Seeded_cfi import *
from ..modules.hltParticleFlowRecHitHGCL1Seeded_cfi import *
from ..modules.hltTiclLayerTileProducerL1Seeded_cfi import *
from ..modules.hltTiclSeedingL1_cfi import *
from ..modules.hltTiclTrackstersCLUE3DHighL1Seeded_cfi import *
from ..modules.hltTiclTracksterLinksL1Seeded_cfi import *
from ..modules.hltBarrelLayerClustersEBL1Seeded_cfi import *
from ..modules.hltTiclEGammaSuperClusterProducerL1Seeded_cfi import hltTiclEGammaSuperClusterProducerL1Seeded
from ..modules.hltTiclTracksterLinksSuperclusteringMustacheL1Seeded_cfi import hltTiclTracksterLinksSuperclusteringMustacheL1Seeded
from ..modules.hltTiclTracksterLinksSuperclusteringDNNL1Seeded_cfi import hltTiclTracksterLinksSuperclusteringDNNL1Seeded


_HgcalLocalRecoL1SeededSequence = cms.Sequence(hltHgcalDigis+
                                               hltL1TEGammaHGCFilteredCollectionProducer+
                                               hltHgcalDigisL1Seeded+
                                               hltHGCalUncalibRecHitL1Seeded+
                                               hltHGCalRecHitL1Seeded+
                                               hltParticleFlowRecHitHGCL1Seeded+
                                               hltRechitInRegionsHGCAL+
                                               hltHgcalLayerClustersEEL1Seeded+
                                               hltHgcalLayerClustersHSciL1Seeded+
                                               hltHgcalLayerClustersHSiL1Seeded+
                                               hltMergeLayerClustersL1Seeded)

_HgcalTICLPatternRecognitionL1SeededSequence = cms.Sequence(hltFilteredLayerClustersCLUE3DHighL1Seeded+
                                                            hltTiclSeedingL1+
                                                            hltTiclLayerTileProducerL1Seeded+
                                                            hltTiclTrackstersCLUE3DHighL1Seeded)



_SuperclusteringL1SeededSequence = cms.Sequence(hltTiclTracksterLinksSuperclusteringDNNL1Seeded
                                                    + hltTiclEGammaSuperClusterProducerL1Seeded)


# The baseline sequence
HLTHgcalTiclPFClusteringForEgammaL1SeededSequence = cms.Sequence(_HgcalLocalRecoL1SeededSequence + _HgcalTICLPatternRecognitionL1SeededSequence + _SuperclusteringL1SeededSequence)






# Mustache
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl
ticl_superclustering_mustache_ticl.toReplaceWith(_SuperclusteringL1SeededSequence, 
                                                cms.Sequence(
                                                             hltTiclTracksterLinksSuperclusteringMustacheL1Seeded
                                                             + hltTiclEGammaSuperClusterProducerL1Seeded
                                                )
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
_HgcalLocalRecoL1SeededSequence_barrel = cms.Sequence(
    hltHgcalDigis+
    hltL1TEGammaHGCFilteredCollectionProducer+
    hltHgcalDigisL1Seeded+
    hltHGCalUncalibRecHitL1Seeded+
    hltHGCalRecHitL1Seeded+
    hltParticleFlowRecHitHGCL1Seeded+
    hltRechitInRegionsHGCAL+
    hltHgcalLayerClustersEEL1Seeded+
    hltHgcalLayerClustersHSciL1Seeded+
    hltHgcalLayerClustersHSiL1Seeded+
    hltBarrelLayerClustersEBL1Seeded+
    hltMergeLayerClustersL1Seeded
) 
ticl_barrel.toReplaceWith(_HgcalLocalRecoL1SeededSequence, _HgcalLocalRecoL1SeededSequence_barrel)
