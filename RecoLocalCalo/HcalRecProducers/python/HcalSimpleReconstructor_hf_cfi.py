import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.HBHEMethod3Parameters_cfi as method3

hfreco = cms.EDProducer("HcalSimpleReconstructor",
    method3.m3Parameters,
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("hcalDigis"),
    Subdetector = cms.string('HF'),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(4),
    samplesToAdd = cms.int32(2),
    tsFromDB = cms.bool(True)
)
