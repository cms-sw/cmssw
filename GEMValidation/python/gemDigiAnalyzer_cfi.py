import FWCore.ParameterSet.Config as cms


gemDigiAnalyzer = cms.EDAnalyzer("GEMDigiAnalyzer",
    verbosoty = cms.untracked.int32(1),
    inputTagRPC = cms.untracked.InputTag("simMuonRPCDigis"),
    inputTagGEM = cms.untracked.InputTag("simMuonGEMDigis")
)

