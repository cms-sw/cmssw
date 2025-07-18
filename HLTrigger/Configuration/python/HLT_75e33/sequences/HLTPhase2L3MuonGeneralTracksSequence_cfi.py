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
from ..modules.hltPhase2L3MuonGeneralTracks_cfi import *


HLTPhase2L3MuonGeneralTracksSequence = cms.Sequence(
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
    +hltPhase2L3MuonGeneralTracks
    )
