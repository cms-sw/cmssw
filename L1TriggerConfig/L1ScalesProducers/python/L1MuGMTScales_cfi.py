import FWCore.ParameterSet.Config as cms

L1MuGMTScales = cms.ESProducer("L1MuGMTScalesProducer",
                                   minDeltaPhi = cms.double(-0.1963495),
                                   signedPackingDeltaPhi = cms.bool(True),
                                   maxOvlEtaDT = cms.double(1.3),
                                   nbitPackingOvlEtaCSC = cms.int32(4),
                                   scaleReducedEtaDT = cms.vdouble(0.0, 0.22, 0.27, 0.58, 0.77, 0.87, 0.92, 1.24, 1.3),
                                   scaleReducedEtaFwdRPC = cms.vdouble(1.04, 1.24, 1.36, 1.48, 1.61, 1.73, 1.85, 1.97, 2.1),
                                   nbitPackingOvlEtaFwdRPC = cms.int32(4),
                                   nbinsDeltaEta = cms.int32(15),
                                   minOvlEtaCSC = cms.double(0.9),
                                   scaleReducedEtaCSC = cms.vdouble(0.9, 1.06, 1.26, 1.46, 1.66, 1.86, 2.06, 2.26, 2.5),
                                   nbinsOvlEtaFwdRPC = cms.int32(7),
                                   nbitPackingReducedEta = cms.int32(4),
                                   scaleOvlEtaRPC = cms.vdouble(0.72, 0.83, 0.93, 1.04, 1.14, 1.24, 1.36, 1.48),
                                   signedPackingDeltaEta = cms.bool(True),
                                   nbinsOvlEtaDT = cms.int32(7),
                                   offsetDeltaPhi = cms.int32(4),
                                   nbinsReducedEta = cms.int32(8),
                                   nbitPackingDeltaPhi = cms.int32(3),
                                   offsetDeltaEta = cms.int32(7),
                                   nbitPackingOvlEtaBrlRPC = cms.int32(4),
                                   nbinsDeltaPhi = cms.int32(8),
                                   nbinsOvlEtaBrlRPC = cms.int32(7),
                                   minDeltaEta = cms.double(-0.3),
                                   maxDeltaPhi = cms.double(0.1527163),
                                   maxOvlEtaCSC = cms.double(1.25),
                                   scaleReducedEtaBrlRPC = cms.vdouble(0.0, 0.06, 0.25, 0.41, 0.54, 0.7, 0.83, 0.93, 1.04),
                                   nbinsOvlEtaCSC = cms.int32(7),
                                   nbitPackingDeltaEta = cms.int32(4),
                                   maxDeltaEta = cms.double(0.3),
                                   minOvlEtaDT = cms.double(0.73125),
                                   nbitPackingOvlEtaDT = cms.int32(4)
                               )


# foo bar baz
# QRP1qR80It2C1
# oespEL6e11pCX
