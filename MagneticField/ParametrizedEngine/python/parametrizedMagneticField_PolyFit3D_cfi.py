import FWCore.ParameterSet.Config as cms

idealMagneticFieldRecordSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('IdealMagneticFieldRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('PolyFit3D'),
    parameters = cms.PSet(
    #BValue = cms.double(4.01242188708911)
    BValue = cms.double(3.8114)
    ),
    label = cms.untracked.string('')
)


