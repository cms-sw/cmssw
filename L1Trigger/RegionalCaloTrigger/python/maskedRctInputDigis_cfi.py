import FWCore.ParameterSet.Config as cms

maskedRctInputDigis = cms.EDProducer("MaskedRctInputDigiProducer",
    hcalDigisLabel = cms.InputTag("hcalTriggerPrimitiveDigis"),
    maskFile = cms.FileInPath('L1Trigger/RegionalCaloTrigger/test/data/emptyMask.txt'),
    useEcal = cms.bool(True),
    useHcal = cms.bool(True),
    ecalDigisLabel = cms.InputTag("ecalTriggerPrimitiveDigis")
)



# foo bar baz
# lY0QxngHK2z3y
# Lc6VVQQEsFQQI
