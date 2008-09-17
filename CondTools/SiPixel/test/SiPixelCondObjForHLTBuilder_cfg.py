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
    firstRun = cms.untracked.uint32(1),
    lastRun = cms.untracked.uint32(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint32(1)
)

process.SiPixelCondObjForHLTBuilder = cms.EDFilter("SiPixelCondObjForHLTBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.0),
    meanGain = cms.double(0.4),
    meanPed = cms.double(50.0),
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationForHLTRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    rmsPed = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(3),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelGainCalibrationForHLTRcd'),
        tag = cms.string('SiPixelGainCalibration_TBuffer_hlt_const')
    )),
    connect = cms.string('sqlite_file:prova.db')
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjForHLTBuilder)


