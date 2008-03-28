import FWCore.ParameterSet.Config as cms

L1MuCSCPtLutRcdSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuCSCPtLutRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


