import FWCore.ParameterSet.Config as cms

L1MuCSCDTLutRcdSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuCSCDTLutRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


