import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    lastRun = cms.untracked.uint32(100),
    timetype = cms.string('runnumber'),
    firstValue= cms.uint64(1),
    lastValue= cms.uint64(1),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRAFT_31X::All"
process.poolDBESSource = cms.ESSource("PoolDBESSource",
                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                      DBParameters = cms.PSet(
    messageLevel = cms.untracked.int32(2),
    authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
                                      timetype = cms.untracked.string('runnumber'),
                                      connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_STRIP'),
                                      toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('SiStripFedCablingRcd'),
    tag = cms.string('SiStripFedCabling_GR09_31X_v1')
    ),
    cms.PSet(
    record = cms.string('SiStripBadChannelRcd'),
    tag = cms.string('SiStripBadChannel_GR09_31X_v1')
    )
    )
                                      )
process.es_prefer = cms.ESPrefer("PoolDBESSource", "poolDBESSource")


# Include masking #

process.siStripQualityESProducer.ListOfRecordToMerge=cms.VPSet(
 cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string(''))
 ,cms.PSet(record=cms.string('SiStripBadChannelRcd'),tag=cms.string(''))
 #,cms.PSet(record=cms.string('SiStripBadModuleRcd' ),tag=cms.string(''))
)
process.siStripQualityESProducer.ReduceGranularity = cms.bool(False)

#-------------------------------------------------
# Services for the TkHistoMap
#-------------------------------------------------
process.load("DQMServices.Core.DQMStore_cfg")
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
#-------------------------------------------------
process.stat = cms.EDAnalyzer("SiStripQualityStatistics",
                              dataLabel = cms.untracked.string(""),
                              SaveTkHistoMap = cms.untracked.bool(True),
                              TkMapFileName = cms.untracked.string("TkMapBadComponents.pdf")  #available filetypes: .pdf .png .jpg .svg
                              )

process.p = cms.Path(process.stat)


