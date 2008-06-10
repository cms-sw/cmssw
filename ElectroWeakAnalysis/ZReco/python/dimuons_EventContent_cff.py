# The following comments couldn't be translated into the new config version:

#    "keep *_goodMuons_*_*",

import FWCore.ParameterSet.Config as cms

dimuonsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep *_selectedLayer1Muons_*_*', 
        'keep *_goodTracks_*_*', 
        'keep *_highPtTracks_*_*', 
        'keep *_goodStandAloneMuonTracks_*_*', 
        'keep *_tkIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_EcalIsolationForTracks_*_*', 
        'keep *_HcalIsolationForTracks_*_*', 
        'keep *_goodMuonIsolations_*_*', 
        'keep *_goodTrackIsolations_*_*', 
        'keep *_muonIsolations_*_*', 
        'keep *_dimuons_*_*', 
        'keep *_dimuonsOneTrack_*_*', 
        'keep *_dimuonsGlobal_*_*', 
        'keep *_dimuonsOneStandAloneMuon_*_*', 
        'keep *_muonMatch_*_*', 
        'keep *_allDimuonsMCMatch_*_*', 
        'keep *_muonHLTMatchHLT1MuonIso_*_*', 
        'keep *_muonHLTMatchHLT1MuonNonIso_*_*', 
        'keep *_muonHLTMatchHLT2MuonNonIso_*_*')
)
dimuonsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('dimuonsPath', 
            'dimuonsOneTrackPath')
    )
)

