import FWCore.ParameterSet.Config as cms

extlut = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuDTExtLutRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

philut = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuDTPhiLutRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

ptalut = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuDTPtaLutRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

etalut = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuDTEtaPatternLutRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

qualut = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuDTQualPatternLutRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

dttfpar = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuDTTFParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

dttfluts = cms.ESProducer("DTTrackFinderConfig")


