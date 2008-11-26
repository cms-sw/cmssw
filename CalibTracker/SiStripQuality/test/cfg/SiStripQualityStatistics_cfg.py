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
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V3P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect='frontier://FrontierProd/CMS_COND_21X_STRIP'
process.poolDBESSource=cms.ESSource("PoolDBESSource",
                                    process.CondDBCommon,
                                    BlobStreamerName=cms.untracked.string('TBufferBlobStreamingService'),
                                    toGet           =cms.VPSet(
    cms.PSet(
    record=cms.string('SiStripBadModuleRcd'),
    tag   =cms.string('SiStripBadChannel_HotStrip_CRAFT_v1_offline')
    )
    )
                                    )

# Include masking #

process.siStripQualityESProducer.ListOfRecordToMerge=cms.VPSet(
 cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string(''))
 ,cms.PSet(record=cms.string('SiStripBadChannelRcd'),tag=cms.string(''))
 ,cms.PSet(record=cms.string('SiStripBadModuleRcd' ),tag=cms.string(''))
)
process.siStripQualityESProducer.ReduceGranularity = cms.bool(False)


process.stat = cms.EDAnalyzer("SiStripQualityStatistics",
                              dataLabel = cms.untracked.string(""),
                              TkMapFileName = cms.untracked.string("TkMapBadComponents.pdf")  #available filetypes: .pdf .png .jpg .svg
                              )

process.p = cms.Path(process.stat)


