import FWCore.ParameterSet.Config as cms

from ..modules.hltHGCalRecHit_cfi import hltHGCalRecHit
from ..modules.hltHGCalUncalibRecHit_cfi import hltHGCalUncalibRecHit
from ..modules.hltHgcalDigis_cfi import hltHgcalDigis
from ..modules.hltHgcalSoALayerClustersProducer_cfi import hltHgcalSoALayerClustersProducer
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import hltHgcalSoARecHitsLayerClustersProducer
from ..modules.hltHgcalSoARecHitsProducer_cfi import hltHgcalSoARecHitsProducer
from ..modules.hltHGCalRecHitL1Seeded_cfi import hltHGCalRecHitL1Seeded
from ..modules.hltHGCalUncalibRecHitL1Seeded_cfi import hltHGCalUncalibRecHitL1Seeded
from ..modules.hltHgcalDigisL1Seeded_cfi import hltHgcalDigisL1Seeded
from ..modules.hltHgcalSoALayerClustersProducerHSci_cfi import hltHgcalSoALayerClustersProducerHSci
from ..modules.hltHgcalSoALayerClustersProducerHSciL1Seeded_cfi import hltHgcalSoALayerClustersProducerHSciL1Seeded
from ..modules.hltHgcalSoALayerClustersProducerHSi_cfi import hltHgcalSoALayerClustersProducerHSi
from ..modules.hltHgcalSoALayerClustersProducerHSiL1Seeded_cfi import hltHgcalSoALayerClustersProducerHSiL1Seeded
from ..modules.hltHgcalSoALayerClustersProducerL1Seeded_cfi import hltHgcalSoALayerClustersProducerL1Seeded
from ..modules.hltHgcalSoARecHitsLayerClustersProducerHSci_cfi import hltHgcalSoARecHitsLayerClustersProducerHSci
from ..modules.hltHgcalSoARecHitsLayerClustersProducerHSciL1Seeded_cfi import hltHgcalSoARecHitsLayerClustersProducerHSciL1Seeded
from ..modules.hltHgcalSoARecHitsLayerClustersProducerHSi_cfi import hltHgcalSoARecHitsLayerClustersProducerHSi
from ..modules.hltHgcalSoARecHitsLayerClustersProducerHSiL1Seeded_cfi import hltHgcalSoARecHitsLayerClustersProducerHSiL1Seeded
from ..modules.hltHgcalSoARecHitsLayerClustersProducerL1Seeded_cfi import hltHgcalSoARecHitsLayerClustersProducerL1Seeded
from ..modules.hltHgcalSoARecHitsProducerHSci_cfi import hltHgcalSoARecHitsProducerHSci
from ..modules.hltHgcalSoARecHitsProducerHSciL1Seeded_cfi import hltHgcalSoARecHitsProducerHSciL1Seeded
from ..modules.hltHgcalSoARecHitsProducerHSi_cfi import hltHgcalSoARecHitsProducerHSi
from ..modules.hltHgcalSoARecHitsProducerHSiL1Seeded_cfi import hltHgcalSoARecHitsProducerHSiL1Seeded
from ..modules.hltHgcalSoARecHitsProducerL1Seeded_cfi import hltHgcalSoARecHitsProducerL1Seeded
from ..modules.hltL1TEGammaHGCFilteredCollectionProducer_cfi import hltL1TEGammaHGCFilteredCollectionProducer
from ..modules.hltRechitInRegionsHGCAL_cfi import hltRechitInRegionsHGCAL
from ..modules.hltInputLST_cfi import hltInputLST
from ..modules.hltInitialStepSeeds_cfi import hltInitialStepSeeds
from ..modules.hltInitialStepTrajectorySeedsLST_cfi import hltInitialStepTrajectorySeedsLST
from ..modules.hltL1GTAcceptFilter_cfi import hltL1GTAcceptFilter
from ..modules.hltLST_cfi import hltLST
from ..modules.hltPhase2OtRecHitsSoA_cfi import hltPhase2OtRecHitsSoA
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
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

#hltExtendedPhase2PixelVerticesSoA = hltPhase2PixelVerticesSoA.clone(pixelTrackSrc = 'hltExtendedPhase2PixelTracksSoA')

HLTLocalTrackerSequence = cms.Sequence(
    hltPhase2SiPixelClustersSoA
    + hltPhase2SiPixelRecHitsSoA
    + hltSiPhase2Clusters
    + hltSiPhase2RecHits
    + hltPhase2OtRecHitsSoA
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
    + hltHgcalSoARecHitsProducerHSi
    + hltHgcalSoARecHitsProducerHSci
    + hltHgcalSoARecHitsLayerClustersProducer
    + hltHgcalSoARecHitsLayerClustersProducerHSi
    + hltHgcalSoARecHitsLayerClustersProducerHSci
    + hltHgcalSoALayerClustersProducer
    + hltHgcalSoALayerClustersProducerHSi
    + hltHgcalSoALayerClustersProducerHSci
)

HLTHeterogeneousHGCalRecoL1SeededSequence = cms.Sequence(
    hltL1TEGammaHGCFilteredCollectionProducer
    + hltHgcalDigisL1Seeded
    + hltHGCalUncalibRecHitL1Seeded
    + hltHGCalRecHitL1Seeded
    + hltRechitInRegionsHGCAL
    + hltHgcalSoARecHitsProducerL1Seeded
    + hltHgcalSoARecHitsProducerHSiL1Seeded
    + hltHgcalSoARecHitsProducerHSciL1Seeded
    + hltHgcalSoARecHitsLayerClustersProducerL1Seeded
    + hltHgcalSoARecHitsLayerClustersProducerHSiL1Seeded
    + hltHgcalSoARecHitsLayerClustersProducerHSciL1Seeded
    + hltHgcalSoALayerClustersProducerL1Seeded
    + hltHgcalSoALayerClustersProducerHSiL1Seeded
    + hltHgcalSoALayerClustersProducerHSciL1Seeded
)

DST_HeterogeneousReco = cms.Path(
    HLTBeginSequence
    + hltL1GTAcceptFilter
    + HLTLocalTrackerSequence
    + HLTPixelTrackingSequence
    + HLTLSTSequence
    + HLTHeterogeneousHGCalRecoSequence
    + HLTHeterogeneousHGCalRecoL1SeededSequence
    + HLTEndSequence
)
