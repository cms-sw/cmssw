import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *

from ..modules.hltHGCalRecHit_cfi import hltHGCalRecHit
from ..modules.hltHGCalUncalibRecHit_cfi import hltHGCalUncalibRecHit
from ..modules.hltHgcalDigis_cfi import hltHgcalDigis
from ..modules.hltHgcalSoALayerClustersProducer_cfi import hltHgcalSoALayerClustersProducer
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import hltHgcalSoARecHitsLayerClustersProducer
from ..modules.hltHgcalSoARecHitsProducer_cfi import hltHgcalSoARecHitsProducer
from ..modules.hltL1GTAcceptFilter_cfi import *
from ..modules.hltPhase2OtRecHitsSoA_cfi import hltPhase2OtRecHitsSoA
from ..modules.hltPhase2PixelRecHitsExtendedSoA_cfi import hltPhase2PixelRecHitsExtendedSoA
from ..modules.hltPhase2PixelVerticesSoA_cfi import hltPhase2PixelVerticesSoA
from ..modules.hltPhase2SiPixelClustersSoA_cfi import hltPhase2SiPixelClustersSoA
from ..modules.hltPhase2SiPixelRecHitsSoA_cfi import hltPhase2SiPixelRecHitsSoA
from ..modules.hltPhase2PixelTracksSoA_cfi import hltPhase2PixelTracksSoA
from ..modules.hltSiPhase2Clusters_cfi import hltSiPhase2Clusters
from ..modules.hltSiPhase2RecHits_cfi import hltSiPhase2RecHits

# this has to come from the auto-generated cfi (no actual physical cfi in the menu yet)
from RecoTracker.PixelSeeding.caHitNtupletAlpakaPhase2OT_cfi import caHitNtupletAlpakaPhase2OT as _hltPhase2PixelTracksSoA

from ..sequences.HLTEndSequence_cfi import *

HLTHeterogeneousTrackSequence = cms.Sequence(hltPhase2SiPixelClustersSoA
                                             + hltPhase2SiPixelRecHitsSoA
                                             + hltPhase2PixelTracksSoA
                                             #+ hltPhase2PixelVerticesSoA # not yet ready
                                             )

# in situ change to get the right rechits and tracks
hltExtendedPhase2PixelTracksSoA = _hltPhase2PixelTracksSoA.clone(pixelRecHitSrc = 'hltPhase2PixelRecHitsExtendedSoA')
hltExtendedPhase2PixelVerticesSoA = hltPhase2PixelVerticesSoA.clone(pixelTrackSrc = 'hltExtendedPhase2PixelTracksSoA')

_HLTHeterogeneousExtendedTrackSequence = cms.Sequence(hltPhase2SiPixelClustersSoA
                                                      + hltPhase2SiPixelRecHitsSoA
                                                      + hltSiPhase2Clusters
                                                      + hltSiPhase2RecHits
                                                      + hltPhase2OtRecHitsSoA
                                                      + hltPhase2PixelRecHitsExtendedSoA
                                                      + hltExtendedPhase2PixelTracksSoA
                                                      #+ hltExtendedPhase2PixelVerticesSoA # not yet ready
                                                      )

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toReplaceWith(HLTHeterogeneousTrackSequence, _HLTHeterogeneousExtendedTrackSequence)

HLTHeterogeneousHGCalRecoSequence = cms.Sequence(hltHgcalDigis +
                                                 hltHGCalUncalibRecHit +
                                                 hltHGCalRecHit +
                                                 hltHgcalSoARecHitsProducer +
                                                 hltHgcalSoARecHitsLayerClustersProducer +
                                                 hltHgcalSoALayerClustersProducer)

DST_HeterogeneousReco = cms.Path(
    HLTBeginSequence
    + hltL1GTAcceptFilter
    + HLTHeterogeneousTrackSequence
    + HLTHeterogeneousHGCalRecoSequence
    + HLTEndSequence
)

# LST Specifics
from ..modules.hltSiPixelClusters_cfi import *
from ..modules.hltSiPixelRecHits_cfi import *
from ..modules.hltPhase2PixelTracks_cfi import *
from ..modules.hltInitialStepSeeds_cfi import *
from ..modules.hltInitialStepSeedTracksLST_cfi import *
from ..modules.hltInputLST_cfi import *
from ..modules.hltLST_cfi import *
from ..modules.hltInitialStepTrajectorySeedsLST_cfi import *

_LSTSequence = cms.Sequence(
    hltSiPixelClusters +
    hltSiPixelRecHits +
    hltSiPhase2Clusters+
    hltSiPhase2RecHits+
    hltPhase2PixelTracks+
    hltInitialStepSeeds+
    hltInitialStepSeedTracksLST+
    hltInputLST+
    hltLST
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toModify(
    DST_HeterogeneousReco,
    lambda path: path.insert(path.index(HLTEndSequence), _LSTSequence)
)
