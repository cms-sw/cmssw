import FWCore.ParameterSet.Config as cms

process = cms.Process("uncalibRecHitsProd")
process.load("EventFilter.EcalTBRawToDigi.ecalTBunpack_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(100000)
        ),
        enable = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        noTimeStamps = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    debugModules = cms.untracked.vstring('ecalTBunpack')
)

process.source = cms.Source("PoolSource",
    maxEvents = cms.untracked.int32(-1),
    fileNames = cms.untracked.vstring('file:/u1/meridian/data/h4/2006/h4b.00011420.A.0.0.root'),
    isBinary = cms.untracked.bool(True)
)

process.EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
    weightsForTB = cms.untracked.bool(False),
    producedEcalPedestals = cms.untracked.bool(True),
    getWeightsFromFile = cms.untracked.bool(False),
    producedEcalWeights = cms.untracked.bool(True),
    producedEcalIntercalibConstants = cms.untracked.bool(True),
    producedEcalGainRatios = cms.untracked.bool(True),
    producedEcalADCToGeVConstant = cms.untracked.bool(True)
)

process.ecalWeightUncalibRecHit = cms.EDProducer("EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.string(''),
    EEhitCollection = cms.string(''),
    EEdigiCollection = cms.string(''),
    digiProducer = cms.string('ecalTBunpack'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('hits.root')
)

process.p = cms.Path(process.ecalTBunpack*process.ecalWeightUncalibRecHit)
process.ep = cms.EndPath(process.out)

