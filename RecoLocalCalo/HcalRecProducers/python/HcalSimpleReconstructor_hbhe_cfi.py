import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.HBHEMethod3Parameters_cfi as method3

hbheprereco = cms.EDProducer("HcalSimpleReconstructor",
    method3.m3Parameters,
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    Subdetector = cms.string('HBHE'),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(4),
    samplesToAdd = cms.int32(4),
    tsFromDB = cms.bool(True)
)
