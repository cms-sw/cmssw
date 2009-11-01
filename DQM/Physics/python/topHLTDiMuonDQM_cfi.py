import FWCore.ParameterSet.Config as cms

topHLTDiMuonDQM = cms.EDAnalyzer("TopHLTDiMuonDQM",

    Level          = cms.untracked.string('L3'),
    TriggerResults = cms.InputTag('TriggerResults','','HLT'),
    hltPaths_L1    = cms.vstring('HLT_L1MuOpen','HLT_L1Mu','HLT_L1Mu20'),
    hltPaths_L3    = cms.vstring('HLT_Mu9','HLT_Mu15','HLT_IsoMu9','HLT_DoubleMu3'),
    L1_Collection  = cms.untracked.InputTag('hltL1extraParticles'),
    L2_Collection  = cms.untracked.InputTag('hltL2MuonCandidates'),
    L3_Collection  = cms.untracked.InputTag('hltL3MuonCandidates'),
    monitorName    = cms.untracked.string('Top/HLTDiMuon')

)

topHLTDiMuonAnalyzer = cms.Sequence(topHLTDiMuonDQM)
