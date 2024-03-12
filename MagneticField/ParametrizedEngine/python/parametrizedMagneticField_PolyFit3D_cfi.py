import FWCore.ParameterSet.Config as cms

idealMagneticFieldRecordSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('IdealMagneticFieldRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

ParametrizedMagneticFieldProducer = cms.ESProducer("ParametrizedMagneticFieldProducer",
    version = cms.string('PolyFit3D'),
    parameters = cms.PSet(
    #BValue = cms.double(2.02156567013928)
    #BValue = cms.double(3.51622117206486)
    BValue = cms.double(3.81143026675623)
    #BValue = cms.double(4.01242188708911)
    ),
    label = cms.untracked.string('')
)


# foo bar baz
# jQ2xJvwcOSaU1
# yGDx8cls9UNBb
