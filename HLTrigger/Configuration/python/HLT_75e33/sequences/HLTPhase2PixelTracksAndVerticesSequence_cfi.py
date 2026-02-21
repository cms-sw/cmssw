import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2OtRecHitsSoA_cfi import hltPhase2OtRecHitsSoA
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import hltPhase2PixelFitterByHelixProjections
from ..modules.hltPhase2PixelRecHitsExtendedSoA_cfi import hltPhase2PixelRecHitsExtendedSoA
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import hltPhase2PixelTrackFilterByKinematics
from ..modules.hltPhase2PixelTracks_cfi import hltPhase2PixelTracks
from ..modules.hltPhase2PixelTracksAndHighPtStepTrackingRegions_cfi import hltPhase2PixelTracksAndHighPtStepTrackingRegions
from ..modules.hltPhase2PixelTracksCAExtension_cfi import hltPhase2PixelTracksCAExtension
from ..modules.hltPhase2PixelTracksCutClassifier_cfi import hltPhase2PixelTracksCutClassifier
from ..modules.hltPhase2PixelTracksHitDoublets_cfi import hltPhase2PixelTracksHitDoublets
from ..modules.hltPhase2PixelTracksHitSeeds_cfi import hltPhase2PixelTracksHitSeeds
from ..modules.hltPhase2PixelTracksSeedLayers_cfi import hltPhase2PixelTracksSeedLayers
from ..modules.hltPhase2PixelTracksSoA_cfi import hltPhase2PixelTracksSoA
from ..modules.hltPhase2PixelVertices_cfi import *
from ..sequences.HLTPhase2PixelVertexingSequence_cfi import HLTPhase2PixelVertexingSequence
from ..sequences.HLTBeamSpotSequence_cfi import HLTBeamSpotSequence

HLTPhase2PixelTracksAndVerticesSequence = cms.Sequence(
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

from ..modules.hltPhase2TrimmedPixelVertices_cfi import hltPhase2TrimmedPixelVertices
_HLTPhase2PixelTracksAndVerticesSequenceTrimming = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +hltPhase2OtRecHitsSoA
    +hltPhase2PixelRecHitsExtendedSoA
    +hltPhase2PixelTracksSoA
    +hltPhase2PixelTracksCAExtension
    +HLTPhase2PixelVertexingSequence
    +hltPhase2PixelTracksCutClassifier
    +hltPhase2PixelTracks
    +hltPhase2TrimmedPixelVertices
)

from Configuration.ProcessModifiers.phase2_hlt_vertexTrimming_cff import phase2_hlt_vertexTrimming
phase2_hlt_vertexTrimming.toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequence,
    _HLTPhase2PixelTracksAndVerticesSequenceTrimming
)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
_HLTPhase2PixelTracksAndVerticesSequenceLegacy = cms.Sequence(
    hltPhase2PixelTracksSeedLayers
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions
    +hltPhase2PixelTracksHitDoublets
    +hltPhase2PixelTracksHitSeeds
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)
hltPhase2LegacyTracking.toReplaceWith(HLTPhase2PixelTracksAndVerticesSequence, _HLTPhase2PixelTracksAndVerticesSequenceLegacy)

from Configuration.ProcessModifiers.hltPhase2LegacyTrackingPatatrackQuadsChain_cff import hltPhase2LegacyTrackingPatatrackQuads
_HLTPhase2PixelTracksAndVerticesSequenceLegacyPatatrack = cms.Sequence(
    HLTBeamSpotSequence
    +hltPhase2PixelTracksAndHighPtStepTrackingRegions
    +hltPhase2PixelFitterByHelixProjections
    +hltPhase2PixelTrackFilterByKinematics
    +hltPhase2PixelTracksSoA
    +hltPhase2PixelTracks
    +HLTPhase2PixelVertexingSequence
)
(hltPhase2LegacyTracking & hltPhase2LegacyTrackingPatatrackQuads).toReplaceWith(
    HLTPhase2PixelTracksAndVerticesSequence,
    _HLTPhase2PixelTracksAndVerticesSequenceLegacyPatatrack
)
