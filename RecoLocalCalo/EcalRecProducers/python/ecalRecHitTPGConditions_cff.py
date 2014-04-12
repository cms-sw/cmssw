import FWCore.ParameterSet.Config as cms

# used for the recovery of the trigger towers
tpparams12 = cms.ESSource("EmptyESSource",
        recordName = cms.string('EcalTPGPhysicsConstRcd'),
        iovIsRunNotTime = cms.bool(True),
        firstValid = cms.vuint32(1)
)
