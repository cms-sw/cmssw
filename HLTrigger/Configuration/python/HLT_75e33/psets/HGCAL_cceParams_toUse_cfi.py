import FWCore.ParameterSet.Config as cms

HGCAL_cceParams_toUse = cms.PSet(
    cceParamFine = cms.vdouble(1.5e+15, -3.00394e-17, 0.318083),
    cceParamThick = cms.vdouble(6e+14, -7.96539e-16, 0.251751),
    cceParamThin = cms.vdouble(1.5e+15, -3.09878e-16, 0.211207)
)
