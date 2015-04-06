import FWCore.ParameterSet.Config as cms

caloStage1Digis = cms.EDProducer(
    "L1TStage1Layer2Producer",
    CaloRegions = cms.InputTag("rctUpgradeFormatDigis"),
    CaloEmCands = cms.InputTag("rctUpgradeFormatDigis"),
    FirmwareVersion = cms.uint32(3),  ## 1=HI algo, 2= pp algo (obsolete), 3= pp algo in HW
    tauMaxJetIsolationA = cms.double(0.1), ## tau isolation cut
    conditionsLabel = cms.string("")
)



