import FWCore.ParameterSet.Config as cms

L1MuCSCAlignmentRcdSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuCSCTFAlignmentRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


