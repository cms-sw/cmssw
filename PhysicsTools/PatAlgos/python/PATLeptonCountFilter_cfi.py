import FWCore.ParameterSet.Config as cms

# module to filter on the number of Leptons
countLayer1Leptons = cms.EDFilter("PATLeptonCountFilter",
    countElectrons = cms.bool(True),
    maxNumber = cms.uint32(999999),
    muonSource = cms.InputTag("selectedLayer1Muons"),
    minNumber = cms.uint32(1),
    electronSource = cms.InputTag("selectedLayer1Electrons"),
    tauSource = cms.InputTag("selectedLayer1Taus"),
    countTaus = cms.bool(False),
    countMuons = cms.bool(True)
)


