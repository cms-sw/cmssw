import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
# process.MessageLogger = cms.Service("MessageLogger",
#     cout = cms.untracked.PSet(
#         threshold = cms.untracked.string('INFO')
#     ),
#     destinations = cms.untracked.vstring('cout')
# )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        # 'file:/afs/cern.ch/user/d/demattia/scratch0/TeVEE.root'
        'file:/afs/cern.ch/user/d/demattia/scratch0/photon_163796_162_155685227.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#-------------------------------------------------
# Calibration
#-------------------------------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_42_V14::All'

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
process.stat = cms.EDAnalyzer("TrackHitPositions",
                              dataLabel = cms.untracked.string(""),
                              SaveTkHistoMap = cms.untracked.bool(True),
                              TkMapFileName = cms.untracked.string("TkMapBadComponents.pdf"),  #available filetypes: .pdf .png .jpg .svg
                              PtCut = cms.double(100)
                              )

process.p = cms.Path(process.stat)


