import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the parabolic magnetic
# field engine.

idealMagneticFieldRecordSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('IdealMagneticFieldRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('Parabolic'),
    label = cms.untracked.string(''),
    parameters=cms.PSet()
)

