import FWCore.ParameterSet.Config as cms

HGCAL_noise_heback = cms.PSet(
    doseMap = cms.string(''),
    noise_MIP = cms.double(0.01),
    scaleByDose = cms.bool(False),
    scaleByDoseAlgo = cms.uint32(0)
)
