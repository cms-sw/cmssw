import FWCore.ParameterSet.Config as cms

L1MuCSCLocalPhiLutRcdSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuCSCLocalPhiLutRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


