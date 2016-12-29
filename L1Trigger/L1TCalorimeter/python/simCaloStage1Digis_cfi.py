import FWCore.ParameterSet.Config as cms

simCaloStage1Digis = cms.EDProducer(
    "L1TStage1Layer2Producer",
    CaloRegions = cms.InputTag("simRctUpgradeFormatDigis"),
    CaloEmCands = cms.InputTag("simRctUpgradeFormatDigis"),
    FirmwareVersion = cms.uint32(3),  ## 1=HI algo, 2= pp algo, 3= pp HW algo
    egRelativeJetIsolationBarrelCut = cms.double(0.3), ## eg isolation cut, 0.3 for loose, 0.2 for tight
    egRelativeJetIsolationEndcapCut = cms.double(0.5), ## eg isolation cut, 0.5 for loose, 0.4 for tight
    tauRelativeJetIsolationCut = cms.double(0.1), ## tau isolation cut
    conditionsLabel = cms.string("")
)
