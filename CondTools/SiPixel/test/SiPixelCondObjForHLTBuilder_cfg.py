import FWCore.ParameterSet.Config as cms

process = cms.Process("PEDESTALS")
process.load("Configuration.StandardSequences.MagneticField_cff")

#process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

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

process.SiPixelCondObjForHLTBuilder = cms.EDAnalyzer("SiPixelCondObjForHLTBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.0002),
    noisyFraction = cms.double(0.0002),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.0),
    meanGain = cms.double(0.4),
    meanPed = cms.double(50.0),
    rmsPed = cms.double(0.0),
# separate input for the FPIX. If not entered the default values are used.
    rmsGainFPix = cms.untracked.double(0.14),
    meanGainFPix = cms.untracked.double(2.8),
    meanPedFPix = cms.untracked.double(28.2),
    rmsPedFPix = cms.untracked.double(2.75),
    
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationForHLTRcd'),
    secondRocRowGainOffset = cms.double(0.0),
    fromFile = cms.bool(False),
    secondRocRowPedOffset = cms.double(0.0),
    generateColumns = cms.untracked.bool(True)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(3),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelGainCalibrationForHLTRcd'),
        tag = cms.string('GainCalib_TEST_hlt')
    )),
    connect = cms.string('sqlite_file:prova.db')
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjForHLTBuilder)


