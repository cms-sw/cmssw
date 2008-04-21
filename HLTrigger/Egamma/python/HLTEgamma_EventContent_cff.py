# The following comments couldn't be translated into the new config version:

#electrons

#photons

#high em

#pixels seeds

#electrons SC and track

#tracks for the isolation

#intermediate steps in the tracking

#electrons

#photons

#high em

#pixels seeds

#electrons SC and track

#tracks for the isolation

import FWCore.ParameterSet.Config as cms

# Full Event content
HLTEgamma_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 
        'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1IsoPhotonTrackIsol_*_*', 
        'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 
        'keep *_hltHcalDoubleCone_*_*', 
        'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 
        'keep *_hltL1IsoElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep *_hltCkfL1IsoTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 
        'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 
        'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 
        'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*')
)
HLTEgamma_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 
        'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1IsoPhotonTrackIsol_*_*', 
        'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 
        'keep *_hltHcalDoubleCone_*_*', 
        'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 
        'keep *_hltL1IsoElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*')
)
HLTEgamma_AOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltL1NonIsoRecoEcalCandidate"), cms.InputTag("hltPixelMatchElectronsL1Iso"), cms.InputTag("hltPixelMatchElectronsL1NonIso"), cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"), 
        cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"), cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"), cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter"), cms.InputTag("hltL1IsoDoublePhotonDoubleEtFilter"), cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter"), cms.InputTag("hltL1NonIsoSingleEMHighEtTrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter"), cms.InputTag("hltL1IsoDoubleExclPhotonTrackIsolFilter"), cms.InputTag("hltL1IsoSinglePhotonPrescaledTrackIsolFilter"), cms.InputTag("hltL1IsoSingleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter"), 
        cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleExclElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleElectronZeePMMassFilter"), cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoLargeWindowDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt8TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt10TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt5PixelMatchFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt5TrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTnoIsoSingleElectronEt8TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt18TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt18TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTIsoSingleElectronEt12TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt40TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt30TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt45TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt30TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8PixelMatchFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10PixelMatchFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12PixelMatchFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoDoublePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt40TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoDoublePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronEt15TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoNoTrkIsoDoublePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoNoTrkIsoSinglePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoNoTrkIsoSinglePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoNoTrkIsoSinglePhotonEt25TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter")),
    outputCommands = cms.untracked.vstring()
)

