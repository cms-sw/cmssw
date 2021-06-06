import FWCore.ParameterSet.Config as cms

l1tdeGEMTPGCommon = cms.PSet(
    monitorDir = cms.string("L1TEMU/L1TdeGEMTPG"),
    verbose = cms.bool(False),
    ## when multiple chambers are enabled, order them by station number!
    chambers = cms.vstring("GE11"),
    dataEmul = cms.vstring("data","emul"),
    clusterVars = cms.vstring("size", "pad", "bx"),
    clusterNBin = cms.vuint32(20,384,10),
    clusterMinBin = cms.vdouble(-0.5,-0.5,-4.5),
    clusterMaxBin = cms.vdouble(19.5,383.5,5.5),
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tdeGEMTPG = DQMEDAnalyzer(
    "L1TdeGEMTPG",
    l1tdeGEMTPGCommon,
    data = cms.InputTag("valMuonGEMPadDigiClusters"),
    emul = cms.InputTag("valMuonGEMPadDigiClusters"),
)
