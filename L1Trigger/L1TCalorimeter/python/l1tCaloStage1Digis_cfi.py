import FWCore.ParameterSet.Config as cms

l1tCaloRCTToUpgradeConverter = cms.EDProducer(
    "l1t::L1TCaloRCTToUpgradeConverter",
    regionTag = cms.InputTag("simRctDigis"),
    emTag = cms.InputTag("simRctDigis"))

l1tCaloStage1Digis = cms.EDProducer(
    "l1t::Stage1Layer2Producer",
    CaloRegions = cms.InputTag("l1tCaloRCTToUpgradeConverter"),
    CaloEmCands = cms.InputTag("l1tCaloRCTToUpgradeConverter"),
    FirmwareVersion = cms.uint32(2),  ## 1=HI algo, 2= pp algo
    egRelativeJetIsolationCut = cms.double(0.5), ## eg isolation cut
    tauRelativeJetIsolationCut = cms.double(1.), ## tau isolation cut
    regionETCutForHT = cms.uint32(7),
    regionETCutForMET = cms.uint32(0),
    minGctEtaForSums = cms.int32(4),
    maxGctEtaForSums = cms.int32(17)
)

l1tPhysicalAdder = cms.EDProducer("l1t::PhysicalEtAdder",
                                  InputCollection = cms.InputTag("l1tCaloStage1Digis")
)

l1tCaloUpgradeToGCTConverter = cms.EDProducer("l1t::L1TCaloUpgradeToGCTConverter",
                                              InputCollection = cms.InputTag("l1tPhysicalAdder")
)


