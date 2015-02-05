import FWCore.ParameterSet.Config as cms

TestCaloAlignmentEP = cms.ESProducer("TestCaloAlignmentEP")

testEBAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EBAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testEBAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EBAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testEEAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EEAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testEEAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EEAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testESAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('ESAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testESAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('ESAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testHBAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HBAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testHBAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HBAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testHEAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HEAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testHEAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HEAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testHOAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HOAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testHOAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HOAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testHFAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HFAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testHFAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HFAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testZDCAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('ZDCAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

testZDCAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('ZDCAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


