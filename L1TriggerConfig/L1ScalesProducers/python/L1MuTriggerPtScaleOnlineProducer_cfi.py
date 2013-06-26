import FWCore.ParameterSet.Config as cms

L1MuTriggerPtScaleOnlineProducer = cms.ESProducer("L1MuTriggerPtScaleOnlineProducer",
                                       onlineDB = cms.string("oracle://CMS_OMDS_LB/CMS_TRG_R"),
                                       onlineAuthentication = cms.string("."),
                                       forceGeneration = cms.bool(False),
                                       ignoreVersionMismatch = cms.bool(False),
# Legacy fields: This stuff should ultimately come from OMDS as well, but at the moment
#                we just define it here like for the dummy producers.
                                       nbitPackingPt = cms.int32(5),
				       signedPackingPt = cms.bool(False),
				       nbinsPt = cms.int32(32)
)
