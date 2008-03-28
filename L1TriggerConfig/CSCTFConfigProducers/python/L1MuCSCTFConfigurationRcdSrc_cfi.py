import FWCore.ParameterSet.Config as cms

L1MuCSCTFConfigurationRcdSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuCSCTFConfigurationRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


