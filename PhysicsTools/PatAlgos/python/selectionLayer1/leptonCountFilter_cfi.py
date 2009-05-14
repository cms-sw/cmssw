import FWCore.ParameterSet.Config as cms

# module to filter on the number of Leptons
countLayer1Leptons = cms.EDFilter("PATLeptonCountFilter",
    electronSource = cms.InputTag("cleanLayer1Electrons"),
    muonSource     = cms.InputTag("cleanLayer1Muons"),
    tauSource      = cms.InputTag("cleanLayer1Taus"),
    countElectrons = cms.bool(True),
    countMuons     = cms.bool(True),
    countTaus      = cms.bool(False),
    minNumber = cms.uint32(0),
    maxNumber = cms.uint32(999999),
)


