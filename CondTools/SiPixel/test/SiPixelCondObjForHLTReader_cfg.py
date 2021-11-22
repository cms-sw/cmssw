import FWCore.ParameterSet.Config as cms

process = cms.Process("PixelDBReader")
process.load("FWCore.MessageService.MessageLogger_cfi")

#process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo.root")
                                   )

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:prova.db'
process.CondDB.DBParameters.messageLevel = 2
process.CondDB.DBParameters.authenticationPath = ''

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(10),
    firstRun = cms.untracked.uint32(1)
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(0)
)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelGainCalibrationForHLTRcd'),
        tag = cms.string('GainCalib_TEST_hlt')
    ))
)

process.prefer("PoolDBESSource")
process.SiPixelCondObjForHLTReader = cms.EDAnalyzer("SiPixelCondObjForHLTReader",
    process.SiPixelGainCalibrationServiceParameters,
    maxRangeDeadPixHist = cms.untracked.double(0.001)

)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjForHLTReader)
#process.ep = cms.EndPath(process.print)
