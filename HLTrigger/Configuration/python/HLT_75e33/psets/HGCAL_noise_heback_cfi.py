import FWCore.ParameterSet.Config as cms

HGCAL_noise_heback = cms.PSet(
    doseMap = cms.string(''),
    noise_MIP = cms.double(0.01),
    scaleByDose = cms.bool(False),
    scaleByDoseAlgo = cms.uint32(0),
    scaleByDoseFactor = cms.double(1),
    sipmMap = cms.string(""),
    referenceIdark = cms.double(-1),
    referenceXtalk = cms.double(-1)
)
