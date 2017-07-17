import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("SiStripQualityStatJob")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
options.register ('runNumber',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Run Number")

options.parseArguments()

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    SiStripQualityStatSummary = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
        SiStripQualityStatistics = cms.untracked.PSet(limit=cms.untracked.int32(100000))
    ),
    destinations = cms.untracked.vstring('cout','SiStripQualityStatSummary'),
    categories = cms.untracked.vstring('SiStripQualityStatistics')
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    # The RunInfo for this run is NOT in the globalTag
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# You can get the bad channel records from a GlobalTag or from specific tags using a PoolDBESSource and an ESPrefer

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

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

process.load("Configuration.Geometry.GeometryRecoDB_cff")

# Include masking #

#configure the SiStripQualityESProducer according to your needs. You can change the configuration of the existing one or create a new one with a label

process.siStripQualityESProducer.ListOfRecordToMerge=cms.VPSet(
    cms.PSet(record=cms.string('SiStripDetCablingRcd'),tag=cms.string(''))
    , cms.PSet(record=cms.string('SiStripBadChannelRcd'),tag=cms.string(''))
    , cms.PSet(record=cms.string('SiStripBadModuleRcd' ),tag=cms.string(''))
    , cms.PSet(record=cms.string('SiStripBadFiberRcd'),tag=cms.string(''))
    , cms.PSet(record=cms.string('SiStripBadStripRcd' ),tag=cms.string(''))
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

#process.onlineSiStripQualityProducer = cms.ESProducer("SiStripQualityESProducer",
#   appendToDataLabel = cms.string('OnlineMasking'),
#   PrintDebugOutput = cms.bool(False),
#   PrintDebug = cms.untracked.bool(True),
#   ListOfRecordToMerge = cms.VPSet(cms.PSet(
#       record = cms.string('SiStripBadChannelRcd'),
#       tag = cms.string('')
#   ), 
#       cms.PSet(
#           record = cms.string('SiStripDetCablingRcd'),
#           tag = cms.string('')
#       ), 
#       cms.PSet(
#           record = cms.string('RunInfoRcd'),
#           tag = cms.string('')
#       )),
#   UseEmptyRunInfo = cms.bool(False),
#   ReduceGranularity = cms.bool(True),
#   ThresholdForReducedGranularity = cms.double(0.3)
#)


#-------------------------------------------------
# Services for the TkHistoMap
#-------------------------------------------------
process.load("DQMServices.Core.DQMStore_cfg")
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
#-------------------------------------------------

# be sure that the dataLabel parameter matches with the label of the SiStripQuality object you want to explore
process.stat = cms.EDAnalyzer("SiStripQualityStatistics",
                              dataLabel = cms.untracked.string(""),
                              SaveTkHistoMap = cms.untracked.bool(True),
                              TkMapFileName = cms.untracked.string("TkMapBadComponents.pdf")  #available filetypes: .pdf .png .jpg .svg
                              )

process.p = cms.Path(process.stat)
