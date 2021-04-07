import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

l1tdeCSCTPGCommon = cms.PSet(
    monitorDir = cms.string('L1TEMU/L1TdeCSCTPG'),
    chambers = cms.vstring("ME1a", "ME1b", "ME12", "ME13", "ME21", "ME22",
                           "ME31", "ME32", "ME41", "ME42"),
    alctVars = cms.vstring("quality", "wiregroup", "bx"),
    alctNBin = cms.vuint32(6, 116, 20),
    alctMinBin = cms.vdouble(0, 0, 0),
    alctMaxBin = cms.vdouble(6, 116, 20),
    clctVars = cms.vstring("quality", "halfstrip", "bx",
                           "pattern", "bend", "quartstrip","eightstrip"),
    clctNBin = cms.vuint32(16, 240, 20, 16, 2, 2, 2),
    clctMinBin = cms.vdouble(0, 0, 0, 0, 0, 0, 0),
    clctMaxBin = cms.vdouble(16, 240, 20, 16, 2, 2, 2),
    lctVars = cms.vstring( "quality", "wiregroup", "halfstrip",
                          "bx", "pattern", "bend", "quartstrip","eightstrip"),
    lctNBin = cms.vuint32(16, 116, 240, 20, 16, 2, 2, 2),
    lctMinBin = cms.vdouble(0, 0, 0, 0, 0, 0, 0, 0),
    lctMaxBin = cms.vdouble(16, 116, 240, 20, 16, 2, 2, 2),
)

l1tdeCSCTPG = DQMEDAnalyzer(
    "L1TdeCSCTPG",
    l1tdeCSCTPGCommon,
    dataALCT = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
    emulALCT = cms.InputTag("valCscStage2Digis"),
    dataCLCT = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
    emulCLCT = cms.InputTag("valCscStage2Digis"),
    dataLCT = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
    emulLCT = cms.InputTag("valCscStage2Digis", "MPCSORTED"),
    dataEmul = cms.vstring("data","emul"),
)
