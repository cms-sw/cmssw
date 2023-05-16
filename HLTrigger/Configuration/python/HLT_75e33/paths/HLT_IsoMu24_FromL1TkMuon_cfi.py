import FWCore.ParameterSet.Config as cms

from ..modules.hltL1TkSingleMuFiltered22_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22L3f24QL3pfecalIsoFiltered0p41_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22L3f24QL3pfhcalIsoFiltered0p40_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22L3f24QL3pfhgcalIsoFiltered4p70_cfi import *
from ..modules.hltL3crIsoL1TkSingleMu22L3f24QL3trkIsoRegionalNewFiltered0p07EcalHcalHgcalTrk_cfi import *
from ..modules.hltL3fL1TkSingleMu22L3Filtered24Q_cfi import *
from ..modules.bunchSpacingProducer_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.hgcalLayerClustersHSci_cfi import *
from ..modules.hgcalLayerClustersEE_cfi import *
from ..modules.hgcalLayerClustersHSi_cfi import *
from ..modules.HGCalRecHit_cfi import *
from ..modules.HGCalUncalibRecHit_cfi import *
from ..modules.hltCsc2DRecHits_cfi import *
from ..modules.hltCscSegments_cfi import *
from ..modules.hltDt1DRecHits_cfi import *
from ..modules.hltDt4DSegments_cfi import *
from ..modules.hltEcalDetIdToBeRecovered_cfi import *
from ..modules.hltEcalDigis_cfi import *
from ..modules.hltEcalPreshowerDigis_cfi import *
from ..modules.hltEcalPreshowerRecHit_cfi import *
from ..modules.hltEcalRecHit_cfi import *
from ..modules.hltEcalUncalibRecHit_cfi import *
from ..modules.hltFixedGridRhoFastjetAllCaloForMuons_cfi import *
from ..modules.hltGemRecHits_cfi import *
from ..modules.hltGemSegments_cfi import *
from ..modules.hltHbhereco_cfi import *
from ..modules.hltHcalDigis_cfi import *
from ..modules.hltHfprereco_cfi import *
from ..modules.hltHfreco_cfi import *
from ..modules.hltHoreco_cfi import *
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
from ..modules.hltParticleFlowClusterECALUncorrectedUnseeded_cfi import *
from ..modules.hltParticleFlowClusterECALUnseeded_cfi import *
from ..modules.hltParticleFlowClusterHBHEForMuons_cfi import *
from ..modules.hltParticleFlowClusterHCALForMuons_cfi import *
from ..modules.hltParticleFlowClusterPSUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitECALUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitHBHEForMuons_cfi import *
from ..modules.hltParticleFlowRecHitPSUnseeded_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelLayerQuadruplets_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelTracks_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelTracksHitDoublets_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelTracksHitQuadruplets_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelTracksTrackingRegions_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonPixelVertices_cfi import *
from ..modules.hltPhase2L3FromL1TkMuonTrimmedPixelVertices_cfi import *
from ..modules.hltPhase2L3GlbMuon_cfi import *
from ..modules.hltPhase2L3MuonCandidates_cfi import *
from ..modules.hltPhase2L3MuonGeneralTracks_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepClusters_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepHitDoublets_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepHitTriplets_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepSeedLayers_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepSeeds_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepTrackCandidates_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepTrackCutClassifier_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepTrackingRegions_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepTracks_cfi import *
from ..modules.hltPhase2L3MuonHighPtTripletStepTracksSelectionHighPurity_cfi import *
from ..modules.hltPhase2L3MuonInitialStepSeeds_cfi import *
from ..modules.hltPhase2L3MuonInitialStepTrackCandidates_cfi import *
from ..modules.hltPhase2L3MuonInitialStepTrackCutClassifier_cfi import *
from ..modules.hltPhase2L3MuonInitialStepTracks_cfi import *
from ..modules.hltPhase2L3MuonInitialStepTracksSelectionHighPurity_cfi import *
from ..modules.hltPhase2L3MuonMerged_cfi import *
from ..modules.hltPhase2L3MuonPixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2L3MuonPixelTrackFilterByKinematics_cfi import *
from ..modules.hltPhase2L3MuonPixelTracks_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksFilter_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksFitter_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksHitDoublets_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksHitQuadruplets_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksSeedLayers_cfi import *
from ..modules.hltPhase2L3MuonPixelTracksTrackingRegions_cfi import *
from ..modules.hltPhase2L3MuonPixelVertices_cfi import *
from ..modules.hltPhase2L3Muons_cfi import *
from ..modules.hltPhase2L3MuonsEcalIsodR0p3dRVeto0p000_cfi import *
from ..modules.hltPhase2L3MuonsHcalIsodR0p3dRVeto0p000_cfi import *
from ..modules.hltPhase2L3MuonsHgcalLCIsodR0p2dRVetoEM0p00dRVetoHad0p02minEEM0p00minEHad0p00_cfi import *
from ..modules.hltPhase2L3MuonsNoID_cfi import *
from ..modules.hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p07_cfi import *
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
from ..modules.hltTowerMakerForAll_cfi import *
from ..modules.MeasurementTrackerEvent_cfi import *
from ..modules.siPhase2Clusters_cfi import *
from ..modules.siPixelClusters_cfi import *
from ..modules.siPixelClusterShapeCache_cfi import *
from ..modules.siPixelRecHits_cfi import *
from ..modules.trackerClusterCheck_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

from ..modules.hgcalMergeLayerClusters_cfi import *

HLT_IsoMu24_FromL1TkMuon = cms.Path(
    HLTBeginSequence +
    hltL1TkSingleMuFiltered22 +
    hltL3fL1TkSingleMu22L3Filtered24Q +
    hltL3crIsoL1TkSingleMu22L3f24QL3pfecalIsoFiltered0p41 +
    hltL3crIsoL1TkSingleMu22L3f24QL3pfhcalIsoFiltered0p40 +
    hltL3crIsoL1TkSingleMu22L3f24QL3pfhgcalIsoFiltered4p70 +
    hltL3crIsoL1TkSingleMu22L3f24QL3trkIsoRegionalNewFiltered0p07EcalHcalHgcalTrk +
    HLTEndSequence,
    cms.Task(
        HGCalRecHit,
        HGCalUncalibRecHit,
        MeasurementTrackerEvent,
        bunchSpacingProducer,
        hgcalDigis,
        hgcalLayerClustersEE,
        hgcalLayerClustersHSi,
        hgcalLayerClustersHSci,
        hgcalMergeLayerClusters,
        hltCsc2DRecHits,
        hltCscSegments,
        hltDt1DRecHits,
        hltDt4DSegments,
        hltEcalDetIdToBeRecovered,
        hltEcalDigis,
        hltEcalPreshowerDigis,
        hltEcalPreshowerRecHit,
        hltEcalRecHit,
        hltEcalUncalibRecHit,
        hltFixedGridRhoFastjetAllCaloForMuons,
        hltGemRecHits,
        hltGemSegments,
        hltHbhereco,
        hltHcalDigis,
        hltHfprereco,
        hltHfreco,
        hltHoreco,
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
        hltParticleFlowClusterECALUncorrectedUnseeded,
        hltParticleFlowClusterECALUnseeded,
        hltParticleFlowClusterHBHEForMuons,
        hltParticleFlowClusterHCALForMuons,
        hltParticleFlowClusterPSUnseeded,
        hltParticleFlowRecHitECALUnseeded,
        hltParticleFlowRecHitHBHEForMuons,
        hltParticleFlowRecHitPSUnseeded,
        hltPhase2L3FromL1TkMuonPixelLayerQuadruplets,
        hltPhase2L3FromL1TkMuonPixelTracks,
        hltPhase2L3FromL1TkMuonPixelTracksHitDoublets,
        hltPhase2L3FromL1TkMuonPixelTracksHitQuadruplets,
        hltPhase2L3FromL1TkMuonPixelTracksTrackingRegions,
        hltPhase2L3FromL1TkMuonPixelVertices,
        hltPhase2L3FromL1TkMuonTrimmedPixelVertices,
        hltPhase2L3GlbMuon,
        hltPhase2L3MuonCandidates,
        hltPhase2L3MuonGeneralTracks,
        hltPhase2L3MuonHighPtTripletStepClusters,
        hltPhase2L3MuonHighPtTripletStepHitDoublets,
        hltPhase2L3MuonHighPtTripletStepHitTriplets,
        hltPhase2L3MuonHighPtTripletStepSeedLayers,
        hltPhase2L3MuonHighPtTripletStepSeeds,
        hltPhase2L3MuonHighPtTripletStepTrackCandidates,
        hltPhase2L3MuonHighPtTripletStepTrackCutClassifier,
        hltPhase2L3MuonHighPtTripletStepTrackingRegions,
        hltPhase2L3MuonHighPtTripletStepTracks,
        hltPhase2L3MuonHighPtTripletStepTracksSelectionHighPurity,
        hltPhase2L3MuonInitialStepSeeds,
        hltPhase2L3MuonInitialStepTrackCandidates,
        hltPhase2L3MuonInitialStepTrackCutClassifier,
        hltPhase2L3MuonInitialStepTracks,
        hltPhase2L3MuonInitialStepTracksSelectionHighPurity,
        hltPhase2L3MuonMerged,
        hltPhase2L3MuonPixelFitterByHelixProjections,
        hltPhase2L3MuonPixelTrackFilterByKinematics,
        hltPhase2L3MuonPixelTracks,
        hltPhase2L3MuonPixelTracksFilter,
        hltPhase2L3MuonPixelTracksFitter,
        hltPhase2L3MuonPixelTracksHitDoublets,
        hltPhase2L3MuonPixelTracksHitQuadruplets,
        hltPhase2L3MuonPixelTracksSeedLayers,
        hltPhase2L3MuonPixelTracksTrackingRegions,
        hltPhase2L3MuonPixelVertices,
        hltPhase2L3MuonTracks,
        hltPhase2L3Muons,
        hltPhase2L3MuonsEcalIsodR0p3dRVeto0p000,
        hltPhase2L3MuonsHcalIsodR0p3dRVeto0p000,
        hltPhase2L3MuonsHgcalLCIsodR0p2dRVetoEM0p00dRVetoHad0p02minEEM0p00minEHad0p00,
        hltPhase2L3MuonsNoID,
        hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p07,
        hltPhase2L3OIL3MuonCandidates,
        hltPhase2L3OIL3Muons,
        hltPhase2L3OIL3MuonsLinksCombination,
        hltPhase2L3OIMuCtfWithMaterialTracks,
        hltPhase2L3OIMuonTrackCutClassifier,
        hltPhase2L3OIMuonTrackSelectionHighPurity,
        hltPhase2L3OISeedsFromL2Muons,
        hltPhase2L3OITrackCandidates,
        hltRpcRecHits,
        hltTowerMakerForAll,
        siPhase2Clusters,
        siPixelClusterShapeCache,
        siPixelClusters,
        siPixelRecHits,
        trackerClusterCheck
    )
)
