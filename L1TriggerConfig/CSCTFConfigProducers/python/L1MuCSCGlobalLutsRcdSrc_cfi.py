import FWCore.ParameterSet.Config as cms

L1MuCSCGlobalLutsRcdSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuCSCGlobalLutsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


