import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "CRAFT_ALL_V5::All"

process.insertNoisyandDead = cms.EDAnalyzer("SiPixelGainCalibrationRejectNoisyAndDead",
    #record = cms.untracked.string('SiPixelGainCalibrationOfflineRcd'),                
    record = cms.untracked.string('SiPixelGainCalibrationForHLTRcd'),                  
    debug = cms.untracked.bool(False)              
)

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )


process.source = cms.Source("EmptyIOVSource",                            
                            lastRun = cms.untracked.uint32(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue=cms.uint64(1),
                            interval = cms.uint64(1)
                            )

#Input DB
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(10),
        authenticationPath = cms.untracked.string('.')
    ),
    toGet = cms.VPSet(cms.PSet(
        #record = cms.string('SiPixelGainCalibrationOfflineRcd'),
        record = cms.string('SiPixelGainCalibrationForHLTRcd'),
        tag = cms.string('SiPixelGainCalibration_TBuffer_hlt_const')
    )),
    connect = cms.string('sqlite_file:prova_HLT.db')

)

#Output DB
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(10),
        authenticationPath = cms.untracked.string('.')
    ),
    toPut = cms.VPSet(
        cms.PSet(
            #record = cms.string('SiPixelGainCalibrationOfflineRcd'),
            record = cms.string('SiPixelGainCalibrationForHLTRcd'),
            tag = cms.string('GainCalib_TEST_hlt')
    )),
    connect = cms.string('sqlite_file:provaOUT.db')
)

process.prefer("PoolDBESSource")
process.p = cms.Path(process.insertNoisyandDead)
