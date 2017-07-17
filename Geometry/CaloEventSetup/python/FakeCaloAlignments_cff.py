import FWCore.ParameterSet.Config as cms

FakeCaloAlignmentEP = cms.ESProducer("FakeCaloAlignmentEP")

fakeEBAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EBAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeEBAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EBAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeEEAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EEAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeEEAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EEAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeESAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('ESAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeESAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('ESAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeHBAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HBAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeHBAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HBAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeHEAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HEAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeHEAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HEAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeHOAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HOAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeHOAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HOAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeHFAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HFAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeHFAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('HFAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeZDCAlignmentSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('ZDCAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

fakeZDCAlignmentErrorSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('ZDCAlignmentErrorExtendedRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


