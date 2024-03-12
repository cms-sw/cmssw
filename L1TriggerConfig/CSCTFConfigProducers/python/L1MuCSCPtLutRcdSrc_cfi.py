import FWCore.ParameterSet.Config as cms

L1MuCSCPtLutRcdSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuCSCPtLutRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


# foo bar baz
# PQ5qqGYIc3isU
# 9MB0XTIBaIl2A
