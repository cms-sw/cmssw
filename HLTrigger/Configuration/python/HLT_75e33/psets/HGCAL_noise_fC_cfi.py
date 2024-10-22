import FWCore.ParameterSet.Config as cms

HGCAL_noise_fC = cms.PSet(
    doseMap = cms.string(''),
    scaleByDose = cms.bool(False),
    scaleByDoseAlgo = cms.uint32(0),
    scaleByDoseFactor = cms.double(1),
    values = cms.vdouble(0.32041011999999996, 0.384492144, 0.32041011999999996)
)