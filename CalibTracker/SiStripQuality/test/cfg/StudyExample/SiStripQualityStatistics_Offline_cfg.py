import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
process.MessageLogger = cms.Service("MessageLogger",
    log_offline = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('log_offline.txt')
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue= cms.uint64(66615),
    lastValue= cms.uint64(70674),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

#-------------------------------------------------
# Calibration
#-------------------------------------------------
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

process.load("CalibTracker.Configuration.Tracker_DependentRecords_forGlobalTag_nofakes_cff")

# Include masking #

process.siStripQualityESProducer.ListOfRecordToMerge=cms.VPSet(
    cms.PSet(record=cms.string('SiStripBadModuleRcd' ),tag=cms.string(''))
)
process.siStripQualityESProducer.ReduceGranularity = cms.bool(False)


process.load("DQM.SiStripMonitorClient.SiStripDQMOnline_cff")
process.DQMStore.referenceFileName = ''
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.stat = cms.EDAnalyzer("SiStripQualityStatistics",
                              dataLabel = cms.untracked.string(""),
                              TkMapFileName = cms.untracked.string("Offline/TkMapBadComponents_Offline.png")  #available filetypes: .pdf .png .jpg .svg
                              )

process.p = cms.Path(process.stat)


