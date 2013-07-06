import FWCore.ParameterSet.Config as cms

process = cms.Process("PixelDBReader")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo.root")
                                   )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:prova.db'
process.CondDBCommon.DBParameters.messageLevel = 2
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.load("CalibTracker.SiPixelESProducers.SiPixelFakeGainESSource_cfi")

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
    process.CondDBCommon,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelGainCalibrationRcd'),
        tag = cms.string('GainCalibTestFull')
    ), 
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationForHLTRcd'),
            tag = cms.string('GainCalibTestHLT')
        ), 
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationOfflineRcd'),
            tag = cms.string('GainCalibTestOffline')
        ))
)

process.prefer("PoolDBESSource")
process.SiPixelCondObjAllPayloadsReader = cms.EDAnalyzer("SiPixelCondObjAllPayloadsReader",
    process.SiPixelGainCalibrationServiceParameters,
    payloadType = cms.string('HLT')
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjAllPayloadsReader)
#process.ep = cms.EndPath(process.print)

