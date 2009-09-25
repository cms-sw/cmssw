import FWCore.ParameterSet.Config as cms

topHLTDiMuonDQM = cms.EDAnalyzer("TopHLTDiMuonDQM",

    Level          = cms.untracked.string('L3'),
    monitorName    = cms.untracked.string('Physics/Top/HLTDiMuon'),

    TriggerResults = cms.InputTag('TriggerResults','','HLT'),
    hltPaths_L1    = cms.vstring('HLT_L1MuOpen','HLT_L1Mu','HLT_L1Mu20'),
    hltPaths_L3    = cms.vstring('HLT_Mu9','HLT_IsoMu9','HLT_Mu15','HLT_DoubleMu3'),
    hltPath_sig    = cms.vstring('HLT_DoubleMu3'),
    hltPath_trig   = cms.vstring('HLT_Mu9'),
    L1_Collection  = cms.untracked.InputTag('hltL1extraParticles'),
    L2_Collection  = cms.untracked.InputTag('hltL2MuonCandidates'),
    L3_Collection  = cms.untracked.InputTag('hltL3MuonCandidates'),

    muon_pT_cut    = cms.double( 10.0 ),
    muon_eta_cut   = cms.double(  2.4 )

)

topHLTDiMuonAnalyzer = cms.Sequence(topHLTDiMuonDQM)
