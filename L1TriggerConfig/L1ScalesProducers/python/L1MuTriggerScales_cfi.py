import FWCore.ParameterSet.Config as cms

L1MuTriggerScales = cms.ESProducer("L1MuTriggerScalesProducer",
                                   nbitPackingDTEta = cms.int32(6),
                                   signedPackingDTEta = cms.bool(False),
                                   nbinsDTEta = cms.int32(64),
                                   minDTEta = cms.double(-1.2),
                                   maxDTEta = cms.double(1.2),
                                   offsetDTEta = cms.int32(0),
                                   nbitPackingCSCEta = cms.int32(6),
                                   nbinsCSCEta = cms.int32(32),
                                   minCSCEta = cms.double(0.9),
                                   maxCSCEta = cms.double(2.5),
                                   scaleRPCEta = cms.vdouble(
          -2.10, -1.97, -1.85, -1.73, -1.61, -1.48,
                -1.36, -1.24, -1.14, -1.04, -0.93, -0.83,
                -0.72, -0.58, -0.44, -0.27, -0.07,
                        0.07,  0.27,  0.44,  0.58,  0.72,
                 0.83,  0.93,  1.04,  1.14,  1.24,  1.36,
                 1.48,  1.61,  1.73,  1.85,  1.97,  2.10),
                                   nbitPackingBrlRPCEta = cms.int32(6),
                                   signedPackingBrlRPCEta = cms.bool(True),
                                   nbinsBrlRPCEta = cms.int32(33),
                                   offsetBrlRPCEta = cms.int32(16),
                                   nbitPackingFwdRPCEta = cms.int32(6),
                                   signedPackingFwdRPCEta = cms.bool(True),
                                   nbinsFwdRPCEta = cms.int32(33),
                                   offsetFwdRPCEta = cms.int32(16),
                                   nbitPackingGMTEta = cms.int32(6),
                                   nbinsGMTEta = cms.int32(31),
                                   scaleGMTEta = cms.vdouble(0.00,
               0.10,  0.20,  0.30,  0.40,  0.50,  0.60,  0.70,  0.80,
               0.90,  1.00,  1.10,  1.20,  1.30,  1.40,  1.50,  1.60,
               1.70,  1.75,  1.80,  1.85,  1.90,  1.95,  2.00,  2.05,
               2.10,  2.15,  2.20,  2.25,  2.30,  2.35,  2.40 ),
                                   nbitPackingPhi = cms.int32(8),
                                   signedPackingPhi = cms.bool(False),
                                   nbinsPhi = cms.int32(144),
                                   minPhi = cms.double(0.),
                                   maxPhi = cms.double(6.2831853)
)


# foo bar baz
# G3RD4UQ8fE4xW
