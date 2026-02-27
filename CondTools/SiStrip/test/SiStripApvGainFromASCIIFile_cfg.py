import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(325642),
    lastValue = cms.uint64(325642),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripApvGainFromFileBuilder=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("DEBUG"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripApvGainFromFileBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

#process.load('Configuration.Geometry.GeometryDB_cff')
#process.load('Configuration.Geometry.GeometryIdeal_cff')

process.load("Configuration.Geometry.GeometryExtended2017_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")

#Setup the SiSTripFedCabling and the SiStripDetCabling
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect='frontier://FrontierProd/CMS_CONDITIONS'

process.poolDBESSource = cms.ESSource('PoolDBESSource',
                                      process.CondDB,
                                      toGet = cms.VPSet( cms.PSet(record = cms.string('SiStripFedCablingRcd'),
                                                                  tag    = cms.string('SiStripFedCabling_GR10_v1_hlt')
                                                                  )
                                                        )
                                     )
                                                                    
process.load("CalibTracker.SiStripESProducers.SiStripConnectivity_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStripApvGainRcd_v1')
    ))
)

from CondTools.SiStrip.siStripApvGainFromFileBuilder_cfi import siStripApvGainFromFileBuilder
process.prod = siStripApvGainFromFileBuilder.clone(doGainNormalization = True,      # do normalize the gains
                                                   putDummyIntoUncabled = True,     # all defects to default
                                                   putDummyIntoUnscanned = True,    # all defects to default
                                                   putDummyIntoOffChannels = True,  # all defects to default
                                                   putDummyIntoBadChannels = True,  # all defects to default
                                                   outputMaps = True,
                                                   outputSummary = True)
process.p = cms.Path(process.prod)


