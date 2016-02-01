import FWCore.ParameterSet.Config as cms

l1tStage2BMTF = cms.EDAnalyzer("L1TStage2BMTF",
                               verbose = cms.untracked.bool(False),
                               stage2bmtfSource = cms.InputTag("bmtfDigis"),
                               DQMStore = cms.untracked.bool(True),
                               monitorDir = cms.untracked.string("L1T2016/L1TStage2BMTF")
                               )
