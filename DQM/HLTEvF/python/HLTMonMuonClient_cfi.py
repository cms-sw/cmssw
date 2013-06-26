import FWCore.ParameterSet.Config as cms

hltmonmuonClient = cms.EDAnalyzer("HLTMonMuonClient",
    input_dir = cms.untracked.string('HLT/HLTMonMuon/Summary'),
    prescaleLS = cms.untracked.int32(-1),
    output_dir = cms.untracked.string('HLT/HLTMonMuon/Client'),
    prescaleEvt = cms.untracked.int32(1)
)

