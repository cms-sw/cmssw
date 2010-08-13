import FWCore.ParameterSet.Config as cms

topHLTDiMuonDQMClient = cms.EDAnalyzer("TopHLTDiMuonDQMClient",

    monitorName = cms.string('HLT/Top/HLTDiMuon/'),

)

topHLTDiMuonClient = cms.Sequence(topHLTDiMuonDQMClient)
