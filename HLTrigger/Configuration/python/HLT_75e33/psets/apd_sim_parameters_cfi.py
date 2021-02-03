import FWCore.ParameterSet.Config as cms

apd_sim_parameters = cms.PSet(
    apdAddToBarrel = cms.bool(False),
    apdDigiTag = cms.string('APD'),
    apdDoPEStats = cms.bool(True),
    apdNonlParms = cms.vdouble(
        1.48, -3.75, 1.81, 1.26, 2.0,
        45, 1.0
    ),
    apdSeparateDigi = cms.bool(True),
    apdShapeTau = cms.double(40.5),
    apdShapeTstart = cms.double(74.5),
    apdSimToPEHigh = cms.double(88200000.0),
    apdSimToPELow = cms.double(2450000.0),
    apdTimeOffWidth = cms.double(0.8),
    apdTimeOffset = cms.double(-13.5)
)