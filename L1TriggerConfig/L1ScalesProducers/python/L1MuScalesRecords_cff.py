import FWCore.ParameterSet.Config as cms

l1muscalesrcd = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuTriggerScalesRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

l1gmtscalesrcd = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuGMTScalesRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


