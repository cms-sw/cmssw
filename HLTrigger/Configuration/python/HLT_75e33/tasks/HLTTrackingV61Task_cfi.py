import FWCore.ParameterSet.Config as cms

from ..modules.MeasurementTrackerEvent_cfi import *
from ..modules.generalTracks_cfi import *
from ..modules.highPtTripletStepClusters_cfi import *
from ..modules.highPtTripletStepHitDoublets_cfi import *
from ..modules.highPtTripletStepHitTriplets_cfi import *
from ..modules.highPtTripletStepSeedLayers_cfi import *
from ..modules.highPtTripletStepSeeds_cfi import *
from ..modules.highPtTripletStepTrackCandidates_cfi import *
from ..modules.highPtTripletStepTrackCutClassifier_cfi import *
from ..modules.highPtTripletStepTrackSelectionHighPurity_cfi import *
from ..modules.hltPhase2PixelTracksAndHighPtStepTrackingRegions_cfi import *
from ..modules.highPtTripletStepTracks_cfi import *
from ..modules.initialStepSeeds_cfi import *
from ..modules.initialStepTrackCandidates_cfi import *
from ..modules.initialStepTrackCutClassifier_cfi import *
from ..modules.initialStepTrackSelectionHighPurity_cfi import *
from ..modules.initialStepTracks_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..modules.hltPhase2PixelTracksHitDoublets_cfi import *
from ..modules.hltPhase2PixelTracksHitSeeds_cfi import *
from ..modules.hltPhase2PixelTracksSeedLayers_cfi import *
from ..modules.pixelTracks_cfi import *
from ..modules.pixelVertices_cfi import *
from ..modules.siPhase2Clusters_cfi import *
from ..modules.siPixelClusterShapeCache_cfi import *
from ..modules.siPixelClusters_cfi import *
from ..modules.siPixelRecHits_cfi import *
from ..modules.trackerClusterCheck_cfi import *
from ..tasks.HLTBeamSpotTask_cfi import *

HLTTrackingV61Task = cms.Task(
    HLTBeamSpotTask,
    MeasurementTrackerEvent,
    generalTracks,
    highPtTripletStepClusters,
    highPtTripletStepHitDoublets,
    highPtTripletStepHitTriplets,
    highPtTripletStepSeedLayers,
    highPtTripletStepSeeds,
    highPtTripletStepTrackCandidates,
    highPtTripletStepTrackCutClassifier,
    highPtTripletStepTrackSelectionHighPurity,
    hltPhase2PixelTracksAndHighPtStepTrackingRegions,
    highPtTripletStepTracks,
    initialStepSeeds,
    initialStepTrackCandidates,
    initialStepTrackCutClassifier,
    initialStepTrackSelectionHighPurity,
    initialStepTracks,
    hltPhase2PixelFitterByHelixProjections,
    hltPhase2PixelTrackFilterByKinematics,
    pixelTracks,
    hltPhase2PixelTracksHitDoublets,
    hltPhase2PixelTracksHitSeeds,
    hltPhase2PixelTracksSeedLayers,
    pixelVertices,
    siPhase2Clusters,
    siPixelClusterShapeCache,
    siPixelClusters,
    siPixelRecHits,
    trackerClusterCheck
)
