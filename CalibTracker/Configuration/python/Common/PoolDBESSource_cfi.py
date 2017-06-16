import FWCore.ParameterSet.Config as cms

from CondCore.CondDB.CondDB_cfi import *
poolDBESSource = cms.ESSource("PoolDBESSource",
    CondDB,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    appendToDataLabel = cms.string(''),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('SiStripFakeNoise')
    ))
)


