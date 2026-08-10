import FWCore.ParameterSet.Config as cms

from ..modules.hltHGCalRecHit_cfi import hltHGCalRecHit
from ..modules.hltHGCalUncalibRecHit_cfi import hltHGCalUncalibRecHit
from ..modules.hltHgcalDigis_cfi import hltHgcalDigis
from ..modules.hltHgcalSoALayerClustersProducer_cfi import hltHgcalSoALayerClustersProducer
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import hltHgcalSoARecHitsLayerClustersProducer
from ..modules.hltHgcalSoARecHitsProducer_cfi import hltHgcalSoARecHitsProducer
from ..modules.hltInputLST_cfi import hltInputLST
from ..modules.hltInitialStepSeeds_cfi import hltInitialStepSeeds
from ..modules.hltInitialStepTrajectorySeedsLST_cfi import hltInitialStepTrajectorySeedsLST
from ..modules.hltL1GTAcceptFilter_cfi import hltL1GTAcceptFilter
from ..modules.hltLST_cfi import hltLST
from ..modules.hltPhase2OtRecHitsSoA_cfi import hltPhase2OtRecHitsSoA
from ..modules.hltPhase2PixelRecHitsExtendedSoA_cfi import hltPhase2PixelRecHitsExtendedSoA
from ..modules.hltPhase2PixelTracks_cfi import hltPhase2PixelTracks
from ..modules.hltPhase2PixelTracksSoA_cfi import hltPhase2PixelTracksSoA
from ..modules.hltPhase2PixelTrackTorchHighPuritySelector_cfi import hltPhase2PixelTrackTorchHighPuritySelector
from ..modules.hltPhase2PixelVertices_cfi import hltPhase2PixelVertices
#from ..modules.hltPhase2PixelVerticesSoA_cfi import hltPhase2PixelVerticesSoA
from ..modules.hltPhase2SiPixelClustersSoA_cfi import hltPhase2SiPixelClustersSoA
from ..modules.hltPhase2SiPixelRecHitsSoA_cfi import hltPhase2SiPixelRecHitsSoA
from ..modules.hltSiPixelClusters_cfi import hltSiPixelClusters
from ..modules.hltSiPixelRecHits_cfi import hltSiPixelRecHits
from ..modules.hltSiPhase2Clusters_cfi import hltSiPhase2Clusters
from ..modules.hltSiPhase2RecHits_cfi import hltSiPhase2RecHits
from ..modules.hltEcalBarrelDigisInRegions_cfi import hltEcalBarrelDigisInRegions
from ..modules.hltEcalDetIdToBeRecovered_cfi import hltEcalDetIdToBeRecovered
from ..modules.hltEcalDigis_cfi import hltEcalDigis
from ..modules.hltEcalMultiFitUncalibRecHit_cfi import hltEcalMultiFitUncalibRecHit
from ..modules.hltElePixelHitTripletsClusterRemoverL1Seeded_cfi import hltElePixelHitTripletsClusterRemoverL1Seeded
from ..modules.hltEleSeedsTrackingRegionsL1Seeded_cfi import hltEleSeedsTrackingRegionsL1Seeded
from ..modules.hltFilteredLayerClustersCLUE3DHighL1Seeded_cfi import hltFilteredLayerClustersCLUE3DHighL1Seeded
from ..modules.hltHcalDigis_cfi import hltHcalDigis
from ..modules.hltHgcalDigisL1Seeded_cfi import hltHgcalDigisL1Seeded
from ..modules.hltHgcalLayerClustersHSciL1Seeded_cfi import hltHgcalLayerClustersHSciL1Seeded
from ..modules.hltHgcalLayerClustersHSiL1Seeded_cfi import hltHgcalLayerClustersHSiL1Seeded
from ..modules.hltL1TEGammaFilteredCollectionProducer_cfi import hltL1TEGammaFilteredCollectionProducer
from ..modules.hltL1TEGammaHGCFilteredCollectionProducer_cfi import hltL1TEGammaHGCFilteredCollectionProducer
from ..modules.hltParticleFlowRecHitECALL1Seeded_cfi import hltParticleFlowRecHitECALL1Seeded
from ..modules.hltPixelLayerTriplets_cfi import hltPixelLayerTriplets
from ..modules.hltTiclLayerTileProducerL1Seeded_cfi import hltTiclLayerTileProducerL1Seeded
from ..modules.hltTiclSeedingL1_cfi import hltTiclSeedingL1
from ..modules.hltTiclTracksterLinksSuperclusteringDNNL1Seeded_cfi import hltTiclTracksterLinksSuperclusteringDNNL1Seeded
from ..modules.hltEcalUncalibRecHitL1Seeded_cfi import hltEcalUncalibRecHitL1Seeded
from ..modules.hltPixelLayerPairsL1Seeded_cfi import hltPixelLayerPairsL1Seeded
from ..modules.hltHbhereco_cfi import hltHbhereco
from ..modules.hltHGCalUncalibRecHitL1Seeded_cfi import hltHGCalUncalibRecHitL1Seeded
from ..modules.hltRechitInRegionsECAL_cfi import hltRechitInRegionsECAL
from ..modules.hltParticleFlowClusterECALUncorrectedL1Seeded_cfi import hltParticleFlowClusterECALUncorrectedL1Seeded
from ..modules.hltElePixelHitDoubletsForTripletsL1Seeded_cfi import hltElePixelHitDoubletsForTripletsL1Seeded
from ..modules.hltEcalRecHit_cfi import hltEcalRecHit
from ..modules.hltTiclTrackstersCLUE3DHighL1Seeded_cfi import hltTiclTrackstersCLUE3DHighL1Seeded
from ..modules.hltEcalRecHitL1Seeded_cfi import hltEcalRecHitL1Seeded
from ..modules.hltElePixelHitDoubletsL1Seeded_cfi import hltElePixelHitDoubletsL1Seeded
from ..modules.hltHGCalRecHitL1Seeded_cfi import hltHGCalRecHitL1Seeded
from ..modules.hltElePixelHitTripletsL1Seeded_cfi import hltElePixelHitTripletsL1Seeded
from ..modules.hltFixedGridRhoFastjetAllCaloForEGamma_cfi import hltFixedGridRhoFastjetAllCaloForEGamma
from ..modules.hltParticleFlowClusterECALL1Seeded_cfi import hltParticleFlowClusterECALL1Seeded
from ..modules.hltElePixelSeedsDoubletsL1Seeded_cfi import hltElePixelSeedsDoubletsL1Seeded
from ..modules.hltRechitInRegionsHGCAL_cfi import hltRechitInRegionsHGCAL
from ..modules.hltElePixelSeedsTripletsL1Seeded_cfi import hltElePixelSeedsTripletsL1Seeded
from ..modules.hltEgammaHoverEL1Seeded_cfi import hltEgammaHoverEL1Seeded
from ..modules.hltParticleFlowSuperClusterECALL1Seeded_cfi import hltParticleFlowSuperClusterECALL1Seeded
from ..modules.hltElePixelSeedsCombinedL1Seeded_cfi import hltElePixelSeedsCombinedL1Seeded
from ..modules.hltHgcalLayerClustersEEL1Seeded_cfi import hltHgcalLayerClustersEEL1Seeded
from ..modules.hltMergeLayerClustersL1Seeded_cfi import hltMergeLayerClustersL1Seeded
from ..modules.hltTiclEGammaSuperClusterProducerL1Seeded_cfi import hltTiclEGammaSuperClusterProducerL1Seeded
from ..modules.hltEgammaCandidatesL1Seeded_cfi import hltEgammaCandidatesL1Seeded
from ..modules.hltEgammaSuperClustersToPixelMatchL1Seeded_cfi import hltEgammaSuperClustersToPixelMatchL1Seeded
from ..modules.hltEgammaElectronPixelSeedsPortable_cfi import hltEgammaElectronPixelSeedsPortable
from ..modules.hltBunchSpacingProducer_cfi import hltBunchSpacingProducer
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

#hltExtendedPhase2PixelVerticesSoA = hltPhase2PixelVerticesSoA.clone(pixelTrackSrc = 'hltExtendedPhase2PixelTracksSoA')

HLTLocalTrackerSequence = cms.Sequence(
    hltPhase2SiPixelClustersSoA
    + hltPhase2SiPixelRecHitsSoA
    + hltSiPhase2Clusters
    + hltSiPhase2RecHits
    + hltPhase2OtRecHitsSoA
    + hltPhase2PixelRecHitsExtendedSoA
    + hltSiPixelClusters
    + hltSiPixelRecHits
)

HLTPixelTrackingSequence = cms.Sequence(
    hltPhase2PixelTracksSoA
    + hltPhase2PixelTrackTorchHighPuritySelector
    + hltPhase2PixelTracks
    #+ hltExtendedPhase2PixelVerticesSoA # not yet ready
)

HLTLSTSequence = cms.Sequence(
    hltInitialStepSeeds
    + hltInputLST
    + hltLST
)

HLTHeterogeneousHGCalRecoSequence = cms.Sequence(
    hltHgcalDigis
    + hltHGCalUncalibRecHit
    + hltHGCalRecHit
    + hltHgcalSoARecHitsProducer
    + hltHgcalSoARecHitsLayerClustersProducer
    + hltHgcalSoALayerClustersProducer
)

HLTHeterogeneousEgammaRecoSequence = cms.Sequence(
    hltL1TEGammaHGCFilteredCollectionProducer
    + hltHgcalDigisL1Seeded
    + hltHGCalUncalibRecHitL1Seeded
    + hltHGCalRecHitL1Seeded
    + hltRechitInRegionsHGCAL
    + hltHgcalLayerClustersEEL1Seeded
    + hltHgcalLayerClustersHSiL1Seeded
    + hltHgcalLayerClustersHSciL1Seeded
    + hltMergeLayerClustersL1Seeded
    + hltTiclLayerTileProducerL1Seeded
    + hltTiclSeedingL1
    + hltFilteredLayerClustersCLUE3DHighL1Seeded
    + hltTiclTrackstersCLUE3DHighL1Seeded
    + hltTiclTracksterLinksSuperclusteringDNNL1Seeded
    + hltTiclEGammaSuperClusterProducerL1Seeded

    + hltEcalDigis
    + hltL1TEGammaFilteredCollectionProducer
    + hltEcalBarrelDigisInRegions
    + hltBunchSpacingProducer
    + hltEcalUncalibRecHitL1Seeded
    + hltEcalDetIdToBeRecovered
    + hltEcalRecHitL1Seeded
    + hltRechitInRegionsECAL
    + hltParticleFlowRecHitECALL1Seeded
    + hltParticleFlowClusterECALUncorrectedL1Seeded
    + hltParticleFlowClusterECALL1Seeded
    + hltParticleFlowSuperClusterECALL1Seeded
    + hltEgammaCandidatesL1Seeded

    + hltHcalDigis
    + hltHbhereco
    + hltEcalMultiFitUncalibRecHit
    + hltEcalRecHit
    + hltFixedGridRhoFastjetAllCaloForEGamma
    + hltEgammaHoverEL1Seeded
    + hltEgammaSuperClustersToPixelMatchL1Seeded

    + hltPixelLayerTriplets
    + hltEleSeedsTrackingRegionsL1Seeded
    + hltElePixelHitDoubletsForTripletsL1Seeded
    + hltElePixelHitTripletsL1Seeded
    + hltElePixelSeedsTripletsL1Seeded

    + hltElePixelHitTripletsClusterRemoverL1Seeded
    + hltPixelLayerPairsL1Seeded
    + hltElePixelHitDoubletsL1Seeded
    + hltElePixelSeedsDoubletsL1Seeded
    + hltElePixelSeedsCombinedL1Seeded

    + hltEgammaElectronPixelSeedsPortable
)

DST_HeterogeneousReco = cms.Path(
    HLTBeginSequence
    + hltL1GTAcceptFilter
    + HLTLocalTrackerSequence
    + HLTPixelTrackingSequence
    + HLTLSTSequence
    + HLTHeterogeneousHGCalRecoSequence
    + HLTHeterogeneousEgammaRecoSequence
    + HLTEndSequence
)
