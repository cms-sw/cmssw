import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptyIOVSource",
                            timetype   = cms.string('timestamp'),
                            #firstValue = cms.uint64(5565740779852738368),
                            #lastValue  = cms.uint64(5565740779852738369),
                            #firstValue = cms.uint64(5566459919991846016),
                            #lastValue = cms.uint64(5566459919991846017),
                            firstValue = cms.uint64(5590608666362382145),
                            lastValue = cms.uint64(5590608666362382146),
                            interval  = cms.uint64(1)
                            )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo_alloffroot")
                                   )

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.Timing = cms.Service("Timing")

process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/GeometryIdeal_cff")


process.GlobalTag.globaltag = 'START311_V1::All'

###### Quality ESProducer --- copied from SiStripESProducer
process.load("CalibTracker.SiPixelESProducers.siPixelQualityESProducer_cfi")

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(
    timetype = cms.untracked.string('timestamp'),
    record = cms.string("SiPixelQualityFromDbRcd"),
    tag = cms.string("SiPixelQuality_v10_mc"),
    connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PIXEL")),
    cms.PSet(timetype = cms.untracked.string('timestamp'),
             record = cms.string("SiPixelDetVOffRcd"),
             tag    = cms.string("SiPixelDetVOff_v1_offline"),    
             connect = cms.untracked.string('sqlite_file:../../../../alloff.db')
             )
    )

process.BadModuleReader = cms.EDAnalyzer("SiPixelBadModuleReader",
                                         printDebug = cms.untracked.uint32(1),
                                         RcdName = cms.untracked.string("SiPixelQualityRcd") 
                                         )

process.siPixelQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet( record = cms.string("SiPixelQualityFromDbRcd"),
              tag    = cms.string("SiPixelQuality_v10_mc")
              ),
    cms.PSet( record = cms.string("SiPixelDetVOffRcd"),
           tag    = cms.string("SiPixelDetVOff_v1_offline")
              )
    )

process.p = cms.Path(process.BadModuleReader)
