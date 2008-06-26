import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *
poolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    appendToDataLabel = cms.string(''),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('SiStripFakeNoise')
    ))
)


