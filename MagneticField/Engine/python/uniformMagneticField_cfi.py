import FWCore.ParameterSet.Config as cms

# This cfi contains everything needed to use the uniform magnetic
# field engine.
idealMagneticFieldRecordSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('IdealMagneticFieldRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

UniformMagneticFieldESProducer = cms.ESProducer("UniformMagneticFieldESProducer",
    ZFieldInTesla = cms.double(0.0),
    label = cms.untracked.string('')
)


