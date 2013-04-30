import FWCore.ParameterSet.Config as cms

process = cms.Process("PixelDBReader")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

# phase1
#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load('Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhaseIPixel_cff')

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo.root")
                                   )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'sqlite_file:prova_all_1440.db'
#process.CondDBCommon.connect = 'sqlite_file:prova_all_1856.db'
#process.CondDBCommon.connect = 'sqlite_file:prova_forhlt_1856.db'
#process.CondDBCommon.connect = 'sqlite_file:prova_offline_1856.db'
process.CondDBCommon.connect = 'sqlite_file:gain_empty_612_slhc1_offline.db'
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
    toGet = cms.VPSet(
##        cms.PSet(
##            record = cms.string('SiPixelGainCalibrationRcd'),
##            tag = cms.string('GainCalibTestFull')
##        ), 
#        cms.PSet(
#            record = cms.string('SiPixelGainCalibrationForHLTRcd'),
#            tag = cms.string('SiPixelGainCalibration_TBuffer_hlt_const')
#            #tag = cms.string('GainCalibTestHLT')
#        ), 
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationOfflineRcd'),
            tag = cms.string('SiPixelGainCalibration_TBuffer_const')
            #tag = cms.string('GainCalibTestOffline')
        ),
    )
)

process.prefer("PoolDBESSource")
process.SiPixelCondObjAllPayloadsReader = cms.EDAnalyzer("SiPixelCondObjAllPayloadsReader",
    process.SiPixelGainCalibrationServiceParameters,
    #payloadType = cms.string('HLT')
    payloadType = cms.string('Offline')
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjAllPayloadsReader)
#process.ep = cms.EndPath(process.print)

