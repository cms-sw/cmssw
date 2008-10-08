import FWCore.ParameterSet.Config as cms

hltmonmuonClient = cms.EDFilter("HLTMonMuonClient",
    input_dir = cms.untracked.string('HLT/HLTMonhltMonMu'),
    prescaleLS = cms.untracked.int32(-1),
    output_dir = cms.untracked.string('HLT/HLTMonhltMonMu/Tests'),
    prescaleEvt = cms.untracked.int32(500)
)

