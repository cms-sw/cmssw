import FWCore.ParameterSet.Config as cms

l1TdeRCT = cms.EDAnalyzer("L1TdeRCT",
    rctSourceData = cms.InputTag("gctDigis"),
    HistFolder = cms.untracked.string('L1TEMU/L1TdeRCT'),
    outputFile = cms.untracked.string(''),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True),
    singlechannelhistos = cms.untracked.bool(False),
    ecalTPGData = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    rctSourceEmul = cms.InputTag("valRctDigis"),
    disableROOToutput = cms.untracked.bool(True),
    hcalTPGData = cms.InputTag("hcalDigis"),
    gtDigisLabel = cms.InputTag("gtDigis"),
    gtEGAlgoName = cms.string("L1_SingleEG1"),
    doubleThreshold = cms.int32(3),
    filterTriggerType = cms.int32(1),
    selectBX= cms.untracked.int32(0)
)

