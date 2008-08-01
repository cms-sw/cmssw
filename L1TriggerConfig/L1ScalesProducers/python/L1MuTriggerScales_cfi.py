import FWCore.ParameterSet.Config as cms

L1MuTriggerScales = cms.ESProducer("L1MuTriggerScalesProducer",
    signedPackingDTEta = cms.bool(True),
    offsetDTEta = cms.int32(32),
    nbinsDTEta = cms.int32(64),
    offsetFwdRPCEta = cms.int32(16),
    signedPackingBrlRPCEta = cms.bool(True),
    maxDTEta = cms.double(1.2),
    nbitPackingFwdRPCEta = cms.int32(6),
    nbinsBrlRPCEta = cms.int32(33),
    minCSCEta = cms.double(0.9),
    nbitPackingGMTEta = cms.int32(6),
    nbinsFwdRPCEta = cms.int32(33),
    nbinsPhi = cms.int32(144),
    nbitPackingPhi = cms.int32(8),
    nbitPackingDTEta = cms.int32(6),
    maxCSCEta = cms.double(2.5),
    nbinsGMTEta = cms.int32(31),
    minDTEta = cms.double(-1.2),
    nbitPackingCSCEta = cms.int32(6),
    signedPackingFwdRPCEta = cms.bool(True),
    offsetBrlRPCEta = cms.int32(16),
    scaleRPCEta = cms.vdouble(-2.1, -1.97, -1.85, -1.73, -1.61, -1.48, -1.36, -1.24, -1.14, -1.04, -0.93, -0.83, -0.72, -0.58, -0.44, -0.27, -0.07, 0.07, 0.27, 0.44, 0.58, 0.72, 0.83, 0.93, 1.04, 1.14, 1.24, 1.36, 1.48, 1.61, 1.73, 1.85, 1.97, 2.1),
    signedPackingPhi = cms.bool(False),
    nbitPackingBrlRPCEta = cms.int32(6),
    nbinsCSCEta = cms.int32(32),
    maxPhi = cms.double(6.2831853),
    minPhi = cms.double(0.0),
    scaleGMTEta = cms.vdouble(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4)
)



