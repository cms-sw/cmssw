import FWCore.ParameterSet.Config as cms

l1CaloGeomRecordSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1CaloGeometryRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


