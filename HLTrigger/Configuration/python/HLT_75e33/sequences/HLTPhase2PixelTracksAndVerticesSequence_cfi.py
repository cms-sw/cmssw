import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..modules.hltPhase2PixelTracks_cfi import *
from ..modules.hltPhase2PixelTracksAndHighPtStepTrackingRegions_cfi import *
from ..modules.hltPhase2PixelTracksHitDoublets_cfi import *
from ..modules.hltPhase2PixelTracksHitSeeds_cfi import *
from ..modules.hltPhase2PixelTracksSeedLayers_cfi import *
from ..sequences.HLTPhase2PixelVertexingSequence_cfi import *

HLTPhase2PixelTracksAndVerticesSequence = cms.Sequence(
    hltPhase2PixelTracksSeedLayers
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions
    +hltPhase2PixelTracksHitDoublets
    +hltPhase2PixelTracksHitSeeds
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)

from ..sequences.HLTBeamSpotSequence_cfi import HLTBeamSpotSequence
from ..modules.hltPhase2PixelTracksSoA_cfi import hltPhase2PixelTracksSoA

_HLTPhase2PixelTracksAndVerticesSequence = cms.Sequence(
   HLTBeamSpotSequence
  +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
  +hltPhase2PixelFitterByHelixProjections # needed by tracker muons
  +hltPhase2PixelTrackFilterByKinematics  # needed by tracker muons
  +hltPhase2PixelTracksSoA
  +hltPhase2PixelTracks
  +HLTPhase2PixelVertexingSequence
)

from ..modules.hltPhase2PixelRecHitsExtendedSoA_cfi import *
from RecoLocalTracker.Phase2TrackerRecHits.phase2OTRecHitsSoAConverter_cfi import *
from ..modules.hltPhase2PixelTracksCAExtension_cfi import hltPhase2PixelTracksCAExtension
from ..modules.hltPhase2PixelTracksCutClassifier_cfi import hltPhase2PixelTracksCutClassifier
_HLTPhase2PixelTracksAndVerticesSequenceCAExtension = cms.Sequence(
   HLTBeamSpotSequence
  +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
  +hltPhase2PixelFitterByHelixProjections # needed by tracker muons
  +hltPhase2PixelTrackFilterByKinematics  # needed by tracker muons
  +phase2OTRecHitsSoAConverter
  +hltPhase2PixelRecHitsExtendedSoA
  +hltPhase2PixelTracksSoA
  +hltPhase2PixelTracksCAExtension
  +HLTPhase2PixelVertexingSequence
  +hltPhase2PixelTracksCutClassifier
  +hltPhase2PixelTracks
)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toReplaceWith(HLTPhase2PixelTracksAndVerticesSequence, _HLTPhase2PixelTracksAndVerticesSequence)

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
(alpaka & phase2CAExtension).toReplaceWith(HLTPhase2PixelTracksAndVerticesSequence, _HLTPhase2PixelTracksAndVerticesSequenceCAExtension)
