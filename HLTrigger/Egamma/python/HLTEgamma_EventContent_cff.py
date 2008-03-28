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
    outputCommands = cms.untracked.vstring('keep *_l1IsoRecoEcalCandidate_*_*', 'keep *_l1NonIsoRecoEcalCandidate_*_*', 'keep *_l1IsolatedElectronHcalIsol_*_*', 'keep *_l1NonIsolatedElectronHcalIsol_*_*', 'keep *_l1IsoElectronTrackIsol_*_*', 'keep *_l1NonIsoElectronTrackIsol_*_*', 'keep *_l1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_l1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_l1IsolatedPhotonEcalIsol_*_*', 'keep *_l1NonIsolatedPhotonEcalIsol_*_*', 'keep *_l1IsolatedPhotonHcalIsol_*_*', 'keep *_l1NonIsolatedPhotonHcalIsol_*_*', 'keep *_l1IsoPhotonTrackIsol_*_*', 'keep *_l1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_l1NonIsoEMHcalDoubleCone_*_*', 'keep *_l1IsoElectronPixelSeeds_*_*', 'keep *_l1NonIsoElectronPixelSeeds_*_*', 'keep *_l1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_l1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_pixelMatchElectronsL1IsoForHLT_*_*', 'keep *_pixelMatchElectronsL1NonIsoForHLT_*_*', 'keep *_pixelMatchElectronsL1IsoLargeWindowForHLT_*_*', 'keep *_pixelMatchElectronsL1NonIsoLargeWindowForHLT_*_*', 'keep *_correctedHybridSuperClustersL1Isolated_*_*', 'keep *_correctedHybridSuperClustersL1NonIsolated_*_*', 'keep *_correctedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_correctedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_ctfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_ctfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_ctfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_ctfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_l1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_l1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_l1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_l1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_l1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_l1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_ckfL1IsoTrackCandidates_*_*', 'keep *_ckfL1NonIsoTrackCandidates_*_*', 'keep *_ckfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_ckfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_l1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_l1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_l1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_l1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_l1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_l1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_l1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_l1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_l1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_l1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_l1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_l1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*')
)
HLTEgamma_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_l1IsoRecoEcalCandidate_*_*', 'keep *_l1NonIsoRecoEcalCandidate_*_*', 'keep *_l1IsolatedElectronHcalIsol_*_*', 'keep *_l1NonIsolatedElectronHcalIsol_*_*', 'keep *_l1IsoElectronTrackIsol_*_*', 'keep *_l1NonIsoElectronTrackIsol_*_*', 'keep *_l1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_l1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_l1IsolatedPhotonEcalIsol_*_*', 'keep *_l1NonIsolatedPhotonEcalIsol_*_*', 'keep *_l1IsolatedPhotonHcalIsol_*_*', 'keep *_l1NonIsolatedPhotonHcalIsol_*_*', 'keep *_l1IsoPhotonTrackIsol_*_*', 'keep *_l1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_l1NonIsoEMHcalDoubleCone_*_*', 'keep *_l1IsoElectronPixelSeeds_*_*', 'keep *_l1NonIsoElectronPixelSeeds_*_*', 'keep *_l1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_l1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_pixelMatchElectronsL1IsoForHLT_*_*', 'keep *_pixelMatchElectronsL1NonIsoForHLT_*_*', 'keep *_pixelMatchElectronsL1IsoLargeWindowForHLT_*_*', 'keep *_pixelMatchElectronsL1NonIsoLargeWindowForHLT_*_*', 'keep *_correctedHybridSuperClustersL1Isolated_*_*', 'keep *_correctedHybridSuperClustersL1NonIsolated_*_*', 'keep *_correctedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_correctedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_ctfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_ctfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_ctfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_l1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_l1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_l1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_l1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_l1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_l1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*')
)
HLTEgamma_AOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("l1IsoRecoEcalCandidate"), cms.InputTag("l1NonIsoRecoEcalCandidate"), cms.InputTag("pixelMatchElectronsL1IsoForHLT"), cms.InputTag("pixelMatchElectronsL1NonIsoForHLT"), cms.InputTag("pixelMatchElectronsL1IsoLargeWindowForHLT"), cms.InputTag("pixelMatchElectronsL1NonIsoLargeWindowForHLT")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter"), cms.InputTag("hltL1IsoDoublePhotonDoubleEtFilter"), cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter"), cms.InputTag("hltL1NonIsoSingleEMHighEtTrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter"), cms.InputTag("hltL1IsoDoubleExclPhotonTrackIsolFilter"), cms.InputTag("hltL1IsoSinglePhotonPrescaledTrackIsolFilter"), cms.InputTag("hltL1IsoSingleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleExclElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleElectronZeePMMassFilter"), cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoLargeWindowDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter")),
    outputCommands = cms.untracked.vstring()
)

