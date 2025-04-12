import FWCore.ParameterSet.Config as cms

PoolDBESSource = cms.ESSource("PoolDBESSource",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelGainCalibrationRcd'),
        tag = cms.string('mytest_p')
    )),
    connect = cms.string('sqlite_file:prova.db'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(1),
        authenticationPath = cms.untracked.string('./'),
        loadBlobStreamer = cms.untracked.bool(True)
    )
)


