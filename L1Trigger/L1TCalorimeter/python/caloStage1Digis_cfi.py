import FWCore.ParameterSet.Config as cms

caloStage1Digis = cms.EDProducer(
    "l1t::Stage1Layer2Producer",
    CaloRegions = cms.InputTag("rctUpgradeFormatDigis"),
    CaloEmCands = cms.InputTag("rctUpgradeFormatDigis"),
    FirmwareVersion = cms.uint32(2),  ## 1=HI algo, 2= pp algo
    egRelativeJetIsolationBarrelCut = cms.double(0.3), ## eg isolation cut, 0.3 for loose, 0.2 for tight
    egRelativeJetIsolationEndcapCut = cms.double(0.5), ## eg isolation cut, 0.5 for loose, 0.4 for tight
    tauRelativeJetIsolationCut = cms.double(1.) ## tau isolation cut
)



