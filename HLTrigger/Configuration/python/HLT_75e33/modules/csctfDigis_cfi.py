import FWCore.ParameterSet.Config as cms

csctfDigis = cms.EDProducer("CSCTFUnpacker",
    MaxBX = cms.int32(11),
    MinBX = cms.int32(5),
    mappingFile = cms.string(''),
    producer = cms.InputTag("rawDataCollector"),
    slot2sector = cms.vint32(
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0
    ),
    swapME1strips = cms.bool(False)
)
