import FWCore.ParameterSet.Config as cms

l1tStage2mGMT = cms.EDAnalyzer("L1TStage2mGMT",
                               verbose = cms.untracked.bool(False),
                               stage2mgmtSource = cms.InputTag("gmtStage2Digis","Muon"),
                               DQMStore = cms.untracked.bool(True),
                               monitorDir = cms.untracked.string("L1T2016/L1TStage2mGMT")
                               )
