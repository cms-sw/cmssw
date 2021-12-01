import FWCore.ParameterSet.Config as cms

omtfFwVersionSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonOverlapFwVersionRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

###OMTF FW ESProducer.
omtfFwVersion = cms.ESProducer(
    "L1TMuonOverlapFwVersionESProducer",
    algoVersion = cms.uint32(0x110),
    layersVersion = cms.uint32(6),
    patternsVersion = cms.uint32(3),
    synthDate = cms.string("2001-01-01 00:00")
)

