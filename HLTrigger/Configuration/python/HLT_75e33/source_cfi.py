import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(
        '/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/TT_TuneCP5_14TeV-powheg-pythia8/FEVT/PU200_111X_mcRun4_realistic_T15_v1-v2/280000/007CCF38-CBE4-6B4D-A97A-580FA0CA0850.root',
    ),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_*_*_RECO',
        'keep *_TTTracksFromTrackletEmulation_Level1TTTracks_RECO',
        'keep *_TTStubsFromPhase2TrackerDigis_*_RECO',
        'keep *_hgcalBackEndLayer2Producer_*_RECO'
    ),
    secondaryFileNames = cms.untracked.vstring()
)
