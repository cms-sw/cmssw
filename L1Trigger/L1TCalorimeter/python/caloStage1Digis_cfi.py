import FWCore.ParameterSet.Config as cms

caloStage1Digis = cms.EDProducer(
    "l1t::Stage1Layer2Producer",
    CaloRegions = cms.InputTag("rctUpgradeFormatDigis"),
    CaloEmCands = cms.InputTag("rctUpgradeFormatDigis"),
    FirmwareVersion = cms.uint32(2),  ## 1=HI algo, 2= pp algo
    egRelativeJetIsolationCut = cms.double(0.5), ## eg isolation cut
    tauRelativeJetIsolationCut = cms.double(1.) ## tau isolation cut
)



