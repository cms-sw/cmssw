import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.HBHEMethod3Parameters_cfi as method3
import RecoLocalCalo.HcalRecProducers.HBHEMethod0Parameters_cfi as method0

hbheprereco = cms.EDProducer("HcalSimpleReconstructor",
    method3.m3Parameters,
    method0.m0Parameters,
    digiLabel = cms.InputTag("hcalDigis"),
    Subdetector = cms.string('HBHE'),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),
    tsFromDB = cms.bool(True)
)

# special M0 settings
hbheprereco.correctionPhaseNS = cms.double(13.0)
hbheprereco.samplesToAdd = cms.int32(4)
