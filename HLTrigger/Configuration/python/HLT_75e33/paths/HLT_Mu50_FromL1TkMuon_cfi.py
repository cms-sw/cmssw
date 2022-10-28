import FWCore.ParameterSet.Config as cms

from ..modules.hltL1TkSingleMuFiltered22_cfi import *
from ..modules.hltL3fL1TkSingleMu22L3Filtered50Q_cfi import *
from ..modules.hltCsc2DRecHits_cfi import *
from ..modules.hltCscSegments_cfi import *
from ..modules.hltDt1DRecHits_cfi import *
from ..modules.hltDt4DSegments_cfi import *
from ..modules.hltGemRecHits_cfi import *
from ..modules.hltGemSegments_cfi import *
from ..modules.hltIter0Phase2L3FromL1TkMuonCkfTrackCandidates_cfi import *
from ..modules.hltIter0Phase2L3FromL1TkMuonCtfWithMaterialTracks_cfi import *
from ..modules.hltIter0Phase2L3FromL1TkMuonPixelSeedsFromPixelTracks_cfi import *
from ..modules.hltIter0Phase2L3FromL1TkMuonTrackCutClassifier_cfi import *
from ..modules.hltIter0Phase2L3FromL1TkMuonTrackSelectionHighPurity_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonCkfTrackCandidates_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonClustersRefRemoval_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonCtfWithMaterialTracks_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonMaskedMeasurementTrackerEvent_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonMerged_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonPixelClusterCheck_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonPixelHitDoublets_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonPixelHitTriplets_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonPixelLayerTriplets_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonPixelSeeds_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonTrackCutClassifier_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonTrackSelectionHighPurity_cfi import *
from ..modules.hltL2MuonFromL1TkMuonCandidates_cfi import *
from ..modules.hltL2MuonSeedsFromL1TkMuon_cfi import *
from ..modules.hltL2MuonsFromL1TkMuon_cfi import *
from ..modules.hltL2OfflineMuonSeeds_cfi import *
from ..modules.hltL3MuonsPhase2L3Links_cfi import *
from ..modules.hltL3MuonsPhase2L3OI_cfi import *
from ..modules.hltMe0RecHits_cfi import *
from ..modules.hltMe0Segments_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelLayerQuadruplets_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelTracks_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelTracksHitDoublets_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelTracksHitQuadruplets_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelTracksTrackingRegions_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelVertices_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonTrimmedPixelVertices_cfi import *
from ..modules.hltPhase2L3GlbMuon_cfi import *
from ..modules.hltPhase2L3MuonCandidates_cfi import *
from ..modules.hltPhase2L3MuonMerged_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksFilter_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksFitter_cfi import *
from ..modules.hltPhase2L3Muons_cfi import *
from ..modules.hltPhase2L3MuonsNoID_cfi import *
from ..modules.hltPhase2L3MuonTracks_cfi import *
from ..modules.hltPhase2L3OIL3MuonCandidates_cfi import *
from ..modules.hltPhase2L3OIL3Muons_cfi import *
from ..modules.hltPhase2L3OIL3MuonsLinksCombination_cfi import *
from ..modules.hltPhase2L3OIMuCtfWithMaterialTracks_cfi import *
from ..modules.hltPhase2L3OIMuonTrackCutClassifier_cfi import *
from ..modules.hltPhase2L3OIMuonTrackSelectionHighPurity_cfi import *
from ..modules.hltPhase2L3OISeedsFromL2Muons_cfi import *
from ..modules.hltPhase2L3OITrackCandidates_cfi import *
from ..modules.hltRpcRecHits_cfi import *
from ..modules.MeasurementTrackerEvent_cfi import *
from ..modules.siPhase2Clusters_cfi import *
from ..modules.siPixelClusters_cfi import *
from ..modules.siPixelClusterShapeCache_cfi import *
from ..modules.siPixelRecHits_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Mu50_FromL1TkMuon = cms.Path(
    HLTBeginSequence +
    hltL1TkSingleMuFiltered22 +
    hltL3fL1TkSingleMu22L3Filtered50Q +
    HLTEndSequence,
    cms.Task(
        MeasurementTrackerEvent,
        hltCsc2DRecHits,
        hltCscSegments,
        hltDt1DRecHits,
        hltDt4DSegments,
        hltGemRecHits,
        hltGemSegments,
        hltIter0Phase2L3FromL1TkMuonCkfTrackCandidates,
        hltIter0Phase2L3FromL1TkMuonCtfWithMaterialTracks,
        hltIter0Phase2L3FromL1TkMuonPixelSeedsFromPixelTracks,
        hltIter0Phase2L3FromL1TkMuonTrackCutClassifier,
        hltIter0Phase2L3FromL1TkMuonTrackSelectionHighPurity,
        hltIter2Phase2L3FromL1TkMuonCkfTrackCandidates,
        hltIter2Phase2L3FromL1TkMuonClustersRefRemoval,
        hltIter2Phase2L3FromL1TkMuonCtfWithMaterialTracks,
        hltIter2Phase2L3FromL1TkMuonMaskedMeasurementTrackerEvent,
        hltIter2Phase2L3FromL1TkMuonMerged,
        hltIter2Phase2L3FromL1TkMuonPixelClusterCheck,
        hltIter2Phase2L3FromL1TkMuonPixelHitDoublets,
        hltIter2Phase2L3FromL1TkMuonPixelHitTriplets,
        hltIter2Phase2L3FromL1TkMuonPixelLayerTriplets,
        hltIter2Phase2L3FromL1TkMuonPixelSeeds,
        hltIter2Phase2L3FromL1TkMuonTrackCutClassifier,
        hltIter2Phase2L3FromL1TkMuonTrackSelectionHighPurity,
        hltL2MuonFromL1TkMuonCandidates,
        hltL2MuonSeedsFromL1TkMuon,
        hltL2MuonsFromL1TkMuon,
        hltL2OfflineMuonSeeds,
        hltL3MuonsPhase2L3Links,
        hltL3MuonsPhase2L3OI,
        hltMe0RecHits,
        hltMe0Segments,
        hltPhase2L3FromL1TkMuonPixelLayerQuadruplets,
        hltPhase2L3FromL1TkMuonPixelTracks,
        hltPhase2L3FromL1TkMuonPixelTracksHitDoublets,
        hltPhase2L3FromL1TkMuonPixelTracksHitQuadruplets,
        hltPhase2L3FromL1TkMuonPixelTracksTrackingRegions,
        hltPhase2L3FromL1TkMuonPixelVertices,
        hltPhase2L3FromL1TkMuonTrimmedPixelVertices,
        hltPhase2L3GlbMuon,
        hltPhase2L3MuonCandidates,
        hltPhase2L3MuonMerged,
        hltPhase2L3MuonPixelTracksFilter,
        hltPhase2L3MuonPixelTracksFitter,
        hltPhase2L3MuonTracks,
        hltPhase2L3Muons,
        hltPhase2L3MuonsNoID,
        hltPhase2L3OIL3MuonCandidates,
        hltPhase2L3OIL3Muons,
        hltPhase2L3OIL3MuonsLinksCombination,
        hltPhase2L3OIMuCtfWithMaterialTracks,
        hltPhase2L3OIMuonTrackCutClassifier,
        hltPhase2L3OIMuonTrackSelectionHighPurity,
        hltPhase2L3OISeedsFromL2Muons,
        hltPhase2L3OITrackCandidates,
        hltRpcRecHits,
        siPhase2Clusters,
        siPixelClusterShapeCache,
        siPixelClusters,
        siPixelRecHits
    )
)
