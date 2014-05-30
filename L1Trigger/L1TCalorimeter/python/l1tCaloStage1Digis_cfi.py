import FWCore.ParameterSet.Config as cms

simRctStage1FormatDigis = cms.EDProducer(
    "l1t::L1TCaloRCTToUpgradeConverter",
    regionTag = cms.InputTag("simRctDigis"),
    emTag = cms.InputTag("simRctDigis"))

simCaloStage1Digis = cms.EDProducer(
    "l1t::Stage1Layer2Producer",
    CaloRegions = cms.InputTag("rctStage1FormatDigis"),
    CaloEmCands = cms.InputTag("rctStage1FormatDigis"),
    FirmwareVersion = cms.uint32(2),  ## 1=HI algo, 2= pp algo
    egRelativeJetIsolationCut = cms.double(0.5), ## eg isolation cut
    tauRelativeJetIsolationCut = cms.double(1.), ## tau isolation cut
    regionETCutForHT = cms.uint32(7),
    regionETCutForMET = cms.uint32(0),
    minGctEtaForSums = cms.int32(4),
    maxGctEtaForSums = cms.int32(17)
)

simCaloStage1FinalDigis = cms.EDProducer("l1t::PhysicalEtAdder",
                                      InputCollection = cms.InputTag("caloStage1Digis")
)

simCaloLegacyFormatDigis = cms.EDProducer("l1t::L1TCaloUpgradeToGCTConverter",
                                       InputCollection = cms.InputTag("caloStage1FinalDigis")
)


