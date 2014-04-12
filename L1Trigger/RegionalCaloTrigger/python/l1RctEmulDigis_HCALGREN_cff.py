import FWCore.ParameterSet.Config as cms

maskedRctInputDigis = cms.EDProducer("MaskedRctInputDigiProducer",
    hcalDigisLabel = cms.InputTag("hcalTriggerPrimitiveDigis"),
    maskFile = cms.FileInPath('L1Trigger/RegionalCaloTrigger/data/maskHcalGren.txt'),
    useEcal = cms.bool(False),
    useHcal = cms.bool(True),
    ecalDigisLabel = cms.InputTag("ecalTriggerPrimitiveDigis")
)

l1RctEmulDigis = cms.EDProducer("L1RCTProducer",
    hcalDigisLabel = cms.InputTag("maskedRctInputDigis"),
    hcalESLabel = cms.string(''),
    ecalESLabel = cms.string(''),
    useEcalCosmicTiming = cms.bool(False),
    postSamples = cms.uint32(0),
    preSamples = cms.uint32(0),
    useHcalCosmicTiming = cms.bool(False),
    useEcal = cms.bool(False),
    useHcal = cms.bool(True),
    ecalDigisLabel = cms.InputTag("maskedRctInputDigis")
)

l1RctSequence = cms.Path(maskedRctInputDigis*l1RctEmulDigis)


