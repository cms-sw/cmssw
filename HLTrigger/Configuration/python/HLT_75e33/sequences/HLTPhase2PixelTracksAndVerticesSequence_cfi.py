import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import hltPhase2PixelFitterByHelixProjections
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import hltPhase2PixelTrackFilterByKinematics
from ..modules.hltPhase2PixelTracks_cfi import hltPhase2PixelTracks
from ..modules.hltPhase2PixelTracksSoA_cfi import hltPhase2PixelTracksSoA
from ..modules.hltPhase2PixelTracksAndHighPtStepTrackingRegions_cfi import hltPhase2PixelTracksAndHighPtStepTrackingRegions
from ..modules.hltPhase2PixelVertices_cfi import *
from ..sequences.HLTPhase2PixelVertexingSequence_cfi import HLTPhase2PixelVertexingSequence
from ..sequences.HLTBeamSpotSequence_cfi import HLTBeamSpotSequence

HLTPhase2PixelTracksAndVerticesSequence = cms.Sequence(
  HLTBeamSpotSequence
  +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
  +hltPhase2PixelFitterByHelixProjections # needed by tracker muons
  +hltPhase2PixelTrackFilterByKinematics  # needed by tracker muons
  +hltPhase2PixelTracksSoA
  +hltPhase2PixelTracks
  +HLTPhase2PixelVertexingSequence
)

from ..modules.hltPhase2PixelRecHitsExtendedSoA_cfi import hltPhase2PixelRecHitsExtendedSoA
from ..modules.hltPhase2OtRecHitsSoA_cfi import hltPhase2OtRecHitsSoA
from ..modules.hltPhase2PixelTracksCAExtension_cfi import hltPhase2PixelTracksCAExtension
from ..modules.hltPhase2PixelTracksCutClassifier_cfi import hltPhase2PixelTracksCutClassifier
_HLTPhase2PixelTracksAndVerticesSequenceCAExtension = cms.Sequence(
   HLTBeamSpotSequence
  +hltPhase2PixelTracksAndHighPtStepTrackingRegions # needed by highPtTripletStep iteration
  +hltPhase2PixelFitterByHelixProjections # needed by tracker muons
  +hltPhase2PixelTrackFilterByKinematics  # needed by tracker muons
  +hltPhase2OtRecHitsSoA
  +hltPhase2PixelRecHitsExtendedSoA
  +hltPhase2PixelTracksSoA
  +hltPhase2PixelTracksCAExtension
  +HLTPhase2PixelVertexingSequence
  +hltPhase2PixelTracksCutClassifier
  +hltPhase2PixelTracks
)

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toReplaceWith(HLTPhase2PixelTracksAndVerticesSequence, _HLTPhase2PixelTracksAndVerticesSequenceCAExtension)
