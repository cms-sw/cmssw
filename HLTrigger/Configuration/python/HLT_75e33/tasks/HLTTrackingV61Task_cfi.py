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
from ..modules.highPtTripletStepTrackingRegions_cfi import *
from ..modules.highPtTripletStepTracks_cfi import *
from ..modules.initialStepSeeds_cfi import *
from ..modules.initialStepTrackCandidates_cfi import *
from ..modules.initialStepTrackCutClassifier_cfi import *
from ..modules.initialStepTrackSelectionHighPurity_cfi import *
from ..modules.initialStepTracks_cfi import *
from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.pixelTrackFilterByKinematics_cfi import *
from ..modules.pixelTracksHitDoublets_cfi import *
from ..modules.pixelTracksHitSeeds_cfi import *
from ..modules.pixelTracksSeedLayers_cfi import *
from ..modules.pixelTracksTrackingRegions_cfi import *
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
    highPtTripletStepTrackingRegions,
    highPtTripletStepTracks,
    initialStepSeeds,
    initialStepTrackCandidates,
    initialStepTrackCutClassifier,
    initialStepTrackSelectionHighPurity,
    initialStepTracks,
    hltPhase2PixelFitterByHelixProjections,
    pixelTrackFilterByKinematics,
    pixelTracks,
    pixelTracksHitDoublets,
    pixelTracksHitSeeds,
    pixelTracksSeedLayers,
    pixelTracksTrackingRegions,
    pixelVertices,
    siPhase2Clusters,
    siPixelClusterShapeCache,
    siPixelClusters,
    siPixelRecHits,
    trackerClusterCheck
)
