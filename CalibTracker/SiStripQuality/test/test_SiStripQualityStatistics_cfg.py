import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("SiStripQualityStatJob")

#prepare options
options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "auto:run2_data",
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.register ('runNumber',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,            # string, int, or float
                  "Run Number")

options.parseArguments()

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripQualityStatistics=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("WARNING"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripQualityStatistics = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    )

# process.MessageLogger = cms.Service("MessageLogger",
#                                     cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING')),
#                                     SiStripQualityStatSummary = cms.untracked.PSet(threshold = cms.untracked.string('INFO'),
#                                                                                    default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                                                    SiStripQualityStatistics = cms.untracked.PSet(limit=cms.untracked.int32(100000))
#                                                                                    ),
#                                     destinations = cms.untracked.vstring('cout','SiStripQualityStatSummary'),                                    
#                                     categories = cms.untracked.vstring('SiStripQualityStatistics')
#                                     )

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

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

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

#-------------------------------------------------
# Services for the TkHistoMap
#-------------------------------------------------
process.load("DQM.SiStripCommon.TkHistoMap_cff")
#-------------------------------------------------

# be sure that the dataLabel parameter matches with the label of the SiStripQuality object you want to explore
from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
process.stat = siStripQualityStatistics.clone(
        StripQualityLabel=cms.string(""),
        SaveTkHistoMap=cms.untracked.bool(False),
        TkMapFileName=cms.untracked.string("TkMapBadComponents.pdf")  #available filetypes: .pdf .png .jpg .svg
        )

process.p = cms.Path(process.stat)
