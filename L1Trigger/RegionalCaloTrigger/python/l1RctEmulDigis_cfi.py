import FWCore.ParameterSet.Config as cms

l1RctEmulDigis = cms.EDProducer("L1RCTProducer",
    hcalDigisLabel = cms.InputTag("hcalTriggerPrimitiveDigis"),
    postSamples = cms.uint32(0),
    preSamples = cms.uint32(0),
    #    FileInPath lutFile = "L1Trigger/RegionalCaloTrigger/data/RCTLUTParameters.txt"
    #    untracked bool orcaFileInput = false       # file below will not be used - but is there for info
    #    FileInPath src = "L1Trigger/RegionalCaloTrigger/data/rct-input-1.dat"
    #    string rctTestInputFile = "-NONE-"
    #    string rctTestOutputFile = "-NONE-"
    #    untracked bool patternTest = false
    #    FileInPath lutFile2 = "L1Trigger/RegionalCaloTrigger/data/RCTHcalScaleFactors.txt"
    useEcal = cms.bool(True),
    useHcal = cms.bool(True),
    ecalDigisLabel = cms.InputTag("ecalTriggerPrimitiveDigis")
)


