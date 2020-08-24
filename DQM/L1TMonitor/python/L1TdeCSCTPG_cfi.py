import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeCSCTPG = DQMEDAnalyzer(
    "L1TdeCSCTPG",
    dataALCT = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
    emulALCT = cms.InputTag("valCscTriggerPrimitiveDigis"),
    dataCLCT = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
    emulCLCT = cms.InputTag("valCscTriggerPrimitiveDigis"),
    dataLCT = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
    emulLCT = cms.InputTag("valCscTriggerPrimitiveDigis", "MPCSORTED"),
    monitorDir = cms.string("L1TEMU/L1TdeCSCTPG"),
    verbose = cms.bool(False),
)
