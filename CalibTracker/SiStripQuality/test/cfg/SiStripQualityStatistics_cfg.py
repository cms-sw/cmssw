import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    # The RunInfo for this run is NOT in the globalTag
    firstValue = cms.uint64(105592),
    lastValue = cms.uint64(105592),
    # The RunInfo for this run is in the globalTag
    # firstValue= cms.uint64(108701),
    # lastValue= cms.uint64(108701),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR09_31X_V4P::All"

# process.poolDBESSource = cms.ESSource("PoolDBESSource",
#                                       BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#                                       DBParameters = cms.PSet(
#     messageLevel = cms.untracked.int32(0),
#     authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#     ),
#                                       timetype = cms.untracked.string('runnumber'),
#                                       connect = cms.string('oracle://cms_orcon_prod/cms_cond_31x_run_info'),
#                                       toGet = cms.VPSet(
#     cms.PSet(
#     record = cms.string('RunInfoRcd'),
#     tag = cms.string('runinfo_start_31X_hlt')
#     ),
#     )
# )
# process.es_prefer = cms.ESPrefer("PoolDBESSource", "poolDBESSource")


# Include masking #

process.siStripQualityESProducer.ListOfRecordToMerge=cms.VPSet(
    cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string(''))
    , cms.PSet(record=cms.string('SiStripBadChannelRcd'),tag=cms.string(''))
    #, cms.PSet(record=cms.string('SiStripBadModuleRcd' ),tag=cms.string(''))
    , cms.PSet(record=cms.string('RunInfoRcd'),tag=cms.string(''))
)
process.siStripQualityESProducer.ReduceGranularity = cms.bool(False)
# True means all the debug output from adding the RunInfo (default is False)
process.siStripQualityESProducer.PrintDebugOutput = cms.bool(True)
# "True" means that the RunInfo is used even if all the feds are off (including other subdetectors).
# This means that if the RunInfo was filled with a fake empty object we will still set the full tracker as bad.
# With "False", instead, in that case the RunInfo information is discarded.
# Default is "False".
process.siStripQualityESProducer.UseEmptyRunInfo = cms.bool(False)

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


