import FWCore.ParameterSet.Config as cms

topHLTDiMuonDQM = cms.EDAnalyzer("TopHLTDiMuonDQM",
    candCollection = cms.untracked.InputTag('hltL1extraParticles'),
    monitorName    = cms.untracked.string('Top/HLTDiMuon')
)

topHLTDiMuonAnalyzer = cms.Sequence(topHLTDiMuonDQM)
