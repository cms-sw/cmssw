import FWCore.ParameterSet.Config as cms

L1MuTriggerScalesOnlineProducer = cms.ESProducer("L1MuTriggerScalesOnlineProducer",
                                       onlineDB = cms.string("oracle://CMS_OMDS_LB/CMS_TRG_R"),
                                       onlineAuthentication = cms.string("."),
                                       forceGeneration = cms.bool(False),
# Legacy fields: This stuff should ultimately come from OMDS as well, but at the moment
#                we just define it here like for the dummy producers.
                                                 
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
                                           # metadata for GMT scales whose contents
                                           # come from DB already
                                           nbitPackingGMTEta = cms.int32(6),
                                           nbinsGMTEta = cms.int32(31),
                                           nbitPackingPhi = cms.int32(8),
                                           signedPackingPhi = cms.bool(False)

)
