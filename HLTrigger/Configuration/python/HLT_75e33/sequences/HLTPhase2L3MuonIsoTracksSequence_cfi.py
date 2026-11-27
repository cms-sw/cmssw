import FWCore.ParameterSet.Config as cms

from ..modules.hltTrackerClusterCheck_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksAndHighPtTripletTrackingRegions_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksSeedLayers_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksHitDoublets_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksHitQuadruplets_cfi import *
from ..modules.hltPhase2L3MuonPixelTracks_cfi import *
from ..modules.hltPhase2L3MuonPixelVertices_cfi import *
from ..modules.hltPhase2L3MuonInitialStepSeeds_cfi import *
from ..modules.hltPhase2L3MuonInitialStepTrackCandidates_cfi import *
from ..modules.hltPhase2L3MuonInitialStepTracks_cfi import *
from ..modules.hltPhase2L3MuonInitialStepTrackCutClassifier_cfi import *
from ..modules.hltPhase2L3MuonInitialStepTracksSelectionHighPurity_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepClusters_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepSeedLayers_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepHitDoublets_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepHitTriplets_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepSeeds_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepTrackCandidates_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepTracks_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepTracksSelectionHighPurity_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepTrackCutClassifier_cfi import *
from ..modules.hltPhase2L3MuonIsoTracks_cfi import *


HLTPhase2L3MuonIsoTracksSequence = cms.Sequence(
    hltTrackerClusterCheck
    +hltPhase2L3MuonPixelTracksAndHighPtTripletTrackingRegions
    +hltPhase2L3MuonPixelTracksSeedLayers
    +hltPhase2L3MuonPixelTracksHitDoublets
    +hltPhase2L3MuonPixelTracksHitQuadruplets
    +hltPhase2L3MuonPixelTracks
    +hltPhase2L3MuonPixelVertices
    +hltPhase2L3MuonInitialStepSeeds
    +hltPhase2L3MuonInitialStepTrackCandidates
    +hltPhase2L3MuonInitialStepTracks
    +hltPhase2L3MuonInitialStepTrackCutClassifier
    +hltPhase2L3MuonInitialStepTracksSelectionHighPurity
    +hltPhase2L3MuonHighPtTripletStepClusters
    +hltPhase2L3MuonHighPtTripletStepSeedLayers
    +hltPhase2L3MuonHighPtTripletStepHitDoublets
    +hltPhase2L3MuonHighPtTripletStepHitTriplets
    +hltPhase2L3MuonHighPtTripletStepSeeds
    +hltPhase2L3MuonHighPtTripletStepTrackCandidates
    +hltPhase2L3MuonHighPtTripletStepTracks
    +hltPhase2L3MuonHighPtTripletStepTrackCutClassifier
    +hltPhase2L3MuonHighPtTripletStepTracksSelectionHighPurity
    +hltPhase2L3MuonIsoTracks
    )

from ..sequences.HLTPhase2PixelTracksAndVerticesSequence_cfi import HLTPhase2PixelTracksAndVerticesSequence
from ..sequences.HLTPhase2PixelVertexingSequence_cfi import HLTPhase2PixelVertexingSequence
# Pixel tracking of the menu, without the pixel vertexing. The copy follows the tracking modifiers.
_HLTPhase2L3MuonIsoTracksSequencePixelSelector = cms.Sequence(
    HLTPhase2PixelTracksAndVerticesSequence.copyAndExclude([HLTPhase2PixelVertexingSequence])
    + hltPhase2L3MuonPixelTracksAndHighPtTripletTrackingRegions
    + hltPhase2L3MuonIsoTracks
)

from Configuration.ProcessModifiers.phase2MuonPixelTracksSelector_cff import phase2MuonPixelTracksSelector
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
from Configuration.ProcessModifiers.phase2MuonSeedsSelector_cff import phase2MuonSeedsSelector
(phase2MuonPixelTracksSelector | ngtScouting | phase2MuonSeedsSelector).toReplaceWith(
    HLTPhase2L3MuonIsoTracksSequence,
    _HLTPhase2L3MuonIsoTracksSequencePixelSelector
)
