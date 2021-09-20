import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(
        '/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/FEVT/PU200_111X_mcRun4_realistic_T15_v1-v2/280000/007CCF38-CBE4-6B4D-A97A-580FA0CA0850.root',
    ),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_*_*_RECO',
        'keep *_pfTracksFromL1TracksBarrel_*_*',
        'keep *_pfTracksFromL1TracksHGCal_*_*',
        'keep *_l1pfCandidates_*_RECO',
        'keep *_TTTracksFromTrackletEmulation_Level1TTTracks_RECO',
        'keep *_TTTracksFromExtendedTrackletEmulation_Level1TTTracks_RECO',
        'keep *_L1TkPrimaryVertex__RECO',
        'keep *_l1pfCandidates_Puppi_RECO',
        'keep *_ak4PFL1Calo__RECO',
        'keep *_ak4PFL1CaloCorrected__RECO',
        'keep *_ak4PFL1PF__RECO',
        'keep *_ak4PFL1PFCorrected__RECO',
        'keep *_ak4PFL1Puppi__RECO',
        'keep *_ak4PFL1PuppiCorrected__RECO',
        'keep *_l1PFMetCalo__RECO',
        'keep *_l1PFMetPF__RECO',
        'keep *_l1PFMetPuppi__RECO',
        'keep *_fixedGridRhoFastjetAll__RECO',
        'keep *_offlineSlimmedPrimaryVertices__RECO',
        'keep *_packedPFCandidates__RECO',
        'keep *_slimmedMuons__RECO',
        'keep *_slimmedJets__RECO',
        'keep *_slimmedJetsPuppi__RECO',
        'keep *_slimmedJetsAK8*_*_RECO',
        'keep *_slimmedMETs__RECO',
        'keep *_slimmedMETsPuppi__RECO'
    ),
    secondaryFileNames = cms.untracked.vstring()
)
