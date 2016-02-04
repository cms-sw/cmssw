import FWCore.ParameterSet.Config as cms

l1demonecal = cms.EDAnalyzer("L1TdeECAL",
    DataEmulCompareSource = cms.InputTag("l1compare"),
    HistFolder = cms.untracked.string('L1TEMU/ECALexpert/'),
    HistFile = cms.untracked.string('l1demon.root'),
    disableROOToutput = cms.untracked.bool(True),
    DQMStore = cms.untracked.bool(True),
    VerboseFlag = cms.untracked.int32(0)
)


