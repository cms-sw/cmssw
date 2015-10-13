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
    pedestalSubtractionType = cms.int32(method3.pedestalSubtractionType),
    pedestalUpperLimit      = cms.double(method3.pedestalUpperLimit),
    timeSlewParsType        = cms.int32(method3.timeSlewParsType),
    timeSlewPars            = cms.vdouble(method3.timeSlewPars),
    respCorrM3              = cms.double(method3.respCorrM3)
)
