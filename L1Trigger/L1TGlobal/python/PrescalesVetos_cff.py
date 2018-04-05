#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

L1TGlobalPrescalesVetosRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TGlobalPrescalesVetosRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

L1TGlobalPrescalesVetos = cms.ESProducer("L1TGlobalPrescalesVetosESProducer",
    TriggerMenuLuminosity = cms.string('startup'),
    Verbosity = cms.int32(0),
    AlgoBxMaskDefault = cms.int32(1),
    PrescaleXMLFile = cms.string('UGT_BASE_RS_PRESCALES_v11.xml'),
    AlgoBxMaskXMLFile = cms.string('UGT_BASE_RS_ALGOBX_MASK_V1.xml'),
    FinOrMaskXMLFile = cms.string('UGT_BASE_RS_FINOR_MASK_v17.xml'),
    VetoMaskXMLFile = cms.string('UGT_BASE_RS_VETO_MASK_v1.xml'),

)


