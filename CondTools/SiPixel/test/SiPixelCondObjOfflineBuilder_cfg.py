import FWCore.ParameterSet.Config as cms

process = cms.Process("PEDESTALS")
process.load("Configuration.StandardSequences.MagneticField_cff")

#process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_31X_V9::All"

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

process.SiPixelCondObjOfflineBuilder = cms.EDAnalyzer("SiPixelCondObjOfflineBuilder",
    process.SiPixelGainCalibrationServiceParameters,
    numberOfModules = cms.int32(2000),
    deadFraction = cms.double(0.00),
    noisyFraction = cms.double(0.00),
    appendMode = cms.untracked.bool(False),
    rmsGain = cms.double(0.14),
    meanGain = cms.double(2.8),
    meanPed = cms.double(28.2),
    rmsPed = cms.double(2.75),
# separate input for the FPIX. If not entered the default values are used.
    rmsGainFPix = cms.untracked.double(0.34),
    meanGainFPix = cms.untracked.double(2.8),
    meanPedFPix = cms.untracked.double(28.2),
    rmsPedFPix = cms.untracked.double(5.75),
                                                
    fileName = cms.string('../macros/phCalibrationFit_C0.dat'),
    record = cms.string('SiPixelGainCalibrationOfflineRcd'),
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
        record = cms.string('SiPixelGainCalibrationOfflineRcd'),
        tag = cms.string('V2_deadpixels_002percent')
    )),
    connect = cms.string('sqlite_file:./prova.db')
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjOfflineBuilder)
#process.ep = cms.EndPath(process.print)


