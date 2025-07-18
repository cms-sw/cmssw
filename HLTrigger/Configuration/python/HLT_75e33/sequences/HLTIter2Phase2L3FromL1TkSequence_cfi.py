import FWCore.ParameterSet.Config as cms

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
from ..modules.hltIter2Phase2L3FromL1TkMuonPixelSeedsFiltered_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonTrackCutClassifier_cfi import *
from ..modules.hltIter2Phase2L3FromL1TkMuonTrackSelectionHighPurity_cfi import *

HLTIter2Phase2L3FromL1TkSequence = cms.Sequence(
    hltIter2Phase2L3FromL1TkMuonClustersRefRemoval
    + hltIter2Phase2L3FromL1TkMuonMaskedMeasurementTrackerEvent
    + hltIter2Phase2L3FromL1TkMuonPixelLayerTriplets
    + hltIter2Phase2L3FromL1TkMuonPixelClusterCheck
    + hltIter2Phase2L3FromL1TkMuonPixelHitDoublets
    + hltIter2Phase2L3FromL1TkMuonPixelHitTriplets
    + hltIter2Phase2L3FromL1TkMuonPixelSeeds
    + hltIter2Phase2L3FromL1TkMuonPixelSeedsFiltered
    + hltIter2Phase2L3FromL1TkMuonCkfTrackCandidates
    + hltIter2Phase2L3FromL1TkMuonCtfWithMaterialTracks
    + hltIter2Phase2L3FromL1TkMuonTrackCutClassifier
    + hltIter2Phase2L3FromL1TkMuonTrackSelectionHighPurity
    + hltIter2Phase2L3FromL1TkMuonMerged
)
