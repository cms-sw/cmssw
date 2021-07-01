import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1TdeRCT = DQMEDAnalyzer('L1TdeRCT',
    rctSourceData = cms.InputTag("gctDigis"),
    HistFolder = cms.untracked.string('L1TEMU/L1TdeRCT'),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True),
    perLSsaving = cms.untracked.bool(False), #driven by DQMServices/Core/python/DQMStore_cfi.py
    singlechannelhistos = cms.untracked.bool(False),
    ecalTPGData = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    rctSourceEmul = cms.InputTag("valRctDigis"),
    hcalTPGData = cms.InputTag("hcalDigis"),
    gtDigisLabel = cms.InputTag("gtDigis"),
    gtEGAlgoName = cms.string("L1_SingleEG1"),
    doubleThreshold = cms.int32(3),
    filterTriggerType = cms.int32(1),
    selectBX= cms.untracked.int32(0)
)

