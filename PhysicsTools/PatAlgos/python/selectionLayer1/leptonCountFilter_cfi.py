import FWCore.ParameterSet.Config as cms

# module to filter on the number of Leptons
countLayer1Leptons = cms.EDFilter("PATLeptonCountFilter",
    electronSource = cms.InputTag("selectedLayer1Electrons"),
    muonSource     = cms.InputTag("selectedLayer1Muons"),
    tauSource      = cms.InputTag("selectedLayer1Taus"),
    countElectrons = cms.bool(True),
    countMuons     = cms.bool(True),
    countTaus      = cms.bool(False),
    minNumber = cms.uint32(0),
    maxNumber = cms.uint32(999999),
)


