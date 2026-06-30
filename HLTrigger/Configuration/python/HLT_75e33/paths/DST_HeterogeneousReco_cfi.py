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
    + hltPhase2PixelVertices
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

DST_HeterogeneousReco = cms.Path(
    HLTBeginSequence
    + hltL1GTAcceptFilter
    + HLTLocalTrackerSequence
    + HLTPixelTrackingSequence
    + HLTLSTSequence
    + HLTHeterogeneousHGCalRecoSequence
    + HLTEndSequence
)
