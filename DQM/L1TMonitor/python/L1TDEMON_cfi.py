import FWCore.ParameterSet.Config as cms

l1demon = cms.EDFilter("L1TDEMON",
    HistFolder = cms.untracked.string('L1TEMU/'),
    HistFile = cms.untracked.string('l1demon.root'),
    disableROOToutput = cms.untracked.bool(True),
    DataEmulCompareSource = cms.InputTag("l1compare"),
    DQMStore = cms.untracked.bool(True),
    VerboseFlag = cms.untracked.int32(0),
)


