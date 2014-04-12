import FWCore.ParameterSet.Config as cms

process = cms.Process("PEDESTALS")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    threshold = cms.untracked.string('INFO')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.SiPixelCondObjBuilder = cms.EDAnalyzer("SiPixelCondObjBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(500),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(5.0),
    meanGain = cms.double(25.0),
    meanPed = cms.double(100.0),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    rmsPed = cms.double(5.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(3),
        authenticationPath = cms.untracked.string('')
    ),
    timetype = cms.string('runnumber'),
    connect = cms.string('sqlite_file:prova.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelGainCalibrationRcd'),
        tag = cms.string('GainCalibTestFull')
    ))
)



