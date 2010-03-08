import FWCore.ParameterSet.Config as cms

l1demongct = cms.EDAnalyzer("L1TdeGCT",
    DataEmulCompareSource = cms.InputTag("l1compare"),
    HistFolder = cms.untracked.string('L1TEMU/GCTexpert/'),
    HistFile = cms.untracked.string(''),
    disableROOToutput = cms.untracked.bool(True),
    DQMStore = cms.untracked.bool(True),
    VerboseFlag = cms.untracked.int32(0)
)


