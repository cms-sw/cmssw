import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripApvGainFromDeDxCalibration=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("DEBUG"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripApvGainFromDeDxCalibration = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )


process.load("Configuration.Geometry.GeometryExtended2024_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")

#Setup the SiSTripFedCabling and the SiStripDetCabling
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect='frontier://FrontierProd/CMS_CONDITIONS'

process.poolDBESSource = cms.ESSource('PoolDBESSource',
                                      process.CondDB,
                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                      toGet = cms.VPSet( cms.PSet(record = cms.string('DeDxCalibrationRcd'),
                                                                  tag    = cms.string('DeDxCalibration_HI_2024_prompt_v2')
                                                                  )
                                                        )
                                     )
                                                                    
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:SiStripApvGainFromDeDxCalibration_HI_2024_prompt_v2.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStripApvGainFromDeDxCalibration_HI_2024_prompt_v2')
    ))
)

from CondTools.SiStrip.siStripApvGainFromDeDxCalibration_cfi import siStripApvGainFromDeDxCalibration
process.prod = siStripApvGainFromDeDxCalibration.clone(
    file = cms.untracked.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    printDebug = cms.untracked.uint32(100)  
)
process.p = cms.Path(process.prod)
