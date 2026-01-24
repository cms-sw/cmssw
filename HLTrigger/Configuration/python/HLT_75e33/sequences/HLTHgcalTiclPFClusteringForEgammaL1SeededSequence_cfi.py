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



# Enable EGammaSuperClusterProducer at HLT in ticl v5
hltTiclTracksterLinksSuperclusteringDNNL1Seeded = hltTiclTracksterLinksL1Seeded.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringDNN"),
        algo_verbosity=cms.int32(0),
        onnxModelPath = cms.FileInPath("RecoHGCal/TICL/data/superclustering/supercls_v3.onnx"),
        nnWorkingPoint=cms.double(0.57247),
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

from RecoHGCal.TICL.ticlEGammaSuperClusterProducer_cfi import ticlEGammaSuperClusterProducer as _ticlEGammaSuperClusterProducer
hltTiclEGammaSuperClusterProducerL1Seeded = _ticlEGammaSuperClusterProducer.clone(
    ticlSuperClusters = "hltTiclTracksterLinksSuperclusteringDNNL1Seeded",
    ticlTrackstersEM = "hltTiclTrackstersCLUE3DHighL1Seeded",
    layerClusters = "hltMergeLayerClustersL1Seeded"
)


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
ticl_superclustering_mustache_ticl.toModify(hltTiclEGammaSuperClusterProducerL1Seeded, 
                                            ticlSuperClusters=cms.InputTag("hltTiclTracksterLinksSuperclusteringMustacheL1Seeded"),
                                            ticlTrackstersEM=cms.InputTag("hltTiclTrackstersCLUE3DHighL1Seeded"),
                                            layerClusters=cms.InputTag("hltMergeLayerClustersL1Seeded"),
                                            enableRegression=cms.bool(False)
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
