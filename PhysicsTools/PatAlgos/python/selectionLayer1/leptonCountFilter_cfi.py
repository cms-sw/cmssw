import FWCore.ParameterSet.Config as cms

# module to filter on the number of Leptons
countPatLeptons = cms.EDFilter("PATLeptonCountFilter",
    electronSource = cms.InputTag("cleanPatElectrons"),
    muonSource     = cms.InputTag("cleanPatMuons"),
    tauSource      = cms.InputTag("cleanPatTaus"),
    countElectrons = cms.bool(True),
    countMuons     = cms.bool(True),
    countTaus      = cms.bool(False),
    minNumber = cms.uint32(0),
    maxNumber = cms.uint32(999999),
)


