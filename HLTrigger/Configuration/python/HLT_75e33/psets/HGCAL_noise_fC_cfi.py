import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase2_hgcalV19_cff import phase2_hgcalV19

HGCAL_noise_fC = cms.PSet(
    doseMap = cms.string(''),
    scaleByDose = cms.bool(False),
    scaleByDoseAlgo = cms.uint32(0),
    scaleByDoseFactor = cms.double(1),
    values = cms.vdouble(0.32041011999999996, 0.384492144, 0.32041011999999996)
)

phase2_hgcalV19.toModify(HGCAL_noise_fC , values = [0.32041011999999996, 0.384492144, 0.32041011999999996, 0.384492144])
