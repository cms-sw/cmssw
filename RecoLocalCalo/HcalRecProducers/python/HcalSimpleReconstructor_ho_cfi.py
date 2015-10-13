import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.HBHEMethod3Parameters_cfi as method3

horeco = cms.EDProducer("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    Subdetector = cms.string('HO'),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(4),
    samplesToAdd = cms.int32(4),
    tsFromDB = cms.bool(True),
    # Configuration parameters for Method 3
    pedestalSubtractionType = method3.pedestalSubtractionType,
    pedestalUpperLimit      = method3.pedestalUpperLimit,
    timeSlewParsType        = method3.timeSlewParsType,
    timeSlewPars            = method3.timeSlewPars,
    respCorrM3              = method3.respCorrM3
)
