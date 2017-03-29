import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("SiStrpDQMQTestTuning")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                                    "DONOTEXIST",
                                    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                                    VarParsing.VarParsing.varType.string,          # string, int, or float
                                    "GlobalTag")
options.register ('dqmFile',
                                    "",
                                    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                                    VarParsing.VarParsing.varType.string,          # string, int, or float
                                    "DQM root file")
options.register ('runNumber',
                                    0,
                                    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                                    VarParsing.VarParsing.varType.int,          # string, int, or float
                                    "run number")

options.parseArguments()

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout','cerr','PCLBadComponents','QTBadModules','TopModulesList'), #Reader, cout
    categories = cms.untracked.vstring('SiStripQualityStatistics',
                                       'BadModuleList',
                                       'TkMapParameters',
                                       'TkMapToBeSaved',
                                       'PSUMapToBeSaved',
				       'TopModules'), #Reader, cout
    debugModules = cms.untracked.vstring('siStripDigis', 
                                         'siStripClusters', 
                                         'siStripZeroSuppression', 
                                         'SiStripClusterizer',
                                         'siStripOfflineAnalyser'),
    cerr = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')
                              ),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO'),
                                default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                TkMapParameters = cms.untracked.PSet(limit=cms.untracked.int32(100000)),
                                TkMapToBeSaved = cms.untracked.PSet(limit=cms.untracked.int32(100000)),
                                PSUMapToBeSaved = cms.untracked.PSet(limit=cms.untracked.int32(100000))
                              ),
    PCLBadComponents = cms.untracked.PSet(threshold = cms.untracked.string('INFO'),
                                default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                SiStripQualityStatistics = cms.untracked.PSet(limit=cms.untracked.int32(100000))
                                ),
    QTBadModules = cms.untracked.PSet(threshold = cms.untracked.string('INFO'),
                                default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                BadModuleList = cms.untracked.PSet(limit=cms.untracked.int32(100000))
                                ),
    TopModulesList = cms.untracked.PSet(threshold = cms.untracked.string('INFO'),
				default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
				TopModules = cms.untracked.PSet(limit=cms.untracked.int32(100000))
				)
                                    
)


## Empty Event Source
process.source = cms.Source("EmptyIOVSource",
                              timetype = cms.string('runnumber'),
                              firstValue= cms.uint64(options.runNumber),
                              lastValue= cms.uint64(options.runNumber),
                              interval = cms.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')
# loading TrackerTopologyEP via GeometryDB (since 62x)
process.load('Configuration.StandardSequences.GeometryDB_cff')
    
# DQM Environment
process.load("DQMServices.Core.DQMStore_cfg")

# SiStrip Offline DQM Client
process.siStripOfflineAnalyser = cms.EDAnalyzer("SiStripOfflineDQM",
       GlobalStatusFilling      = cms.untracked.int32(-1),
#        GlobalStatusFilling      = cms.untracked.int32(2),
        SummaryCreationFrequency  = cms.untracked.int32(-1),                                              
#       CreateSummary            = cms.untracked.bool(False),
       SummaryConfigPath        = cms.untracked.string("DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml"),
       UsedWithEDMtoMEConverter = cms.untracked.bool(False),
       PrintFaultyModuleList    = cms.untracked.bool(False),

      InputFileName            = cms.untracked.string(options.dqmFile),
       OutputFileName           = cms.untracked.string("/tmp/testRunNum.root"), 
       CreateTkMap              = cms.untracked.bool(True),
       TkmapParameters          = cms.untracked.PSet(
          loadFedCabling    = cms.untracked.bool(True),
          trackerdatPath    = cms.untracked.string('CommonTools/TrackerMap/data/'),
          trackermaptxtPath = cms.untracked.string('CommonTools/TrackerMap/data/'),
          mapMin            = cms.untracked.double(0.),
          meanToMaxFact     = cms.untracked.double(2.5)
       ),
       TkMapOptions             = cms.untracked.VPSet(
    cms.PSet(mapName=cms.untracked.string('QTestAlarm'),fedMap=cms.untracked.bool(True),useSSQuality=cms.untracked.bool(True),ssqLabel=cms.untracked.string(""),psuMap=cms.untracked.bool(True),loadLVCabling=cms.untracked.bool(True),mapMax=cms.untracked.double(1.),RunNumber=cms.untracked.uint32(options.runNumber)),
    cms.PSet(mapName=cms.untracked.string('FractionOfBadChannels'),mapMax=cms.untracked.double(-1.),logScale=cms.untracked.bool(True),RunNumber=cms.untracked.uint32(options.runNumber)),
    cms.PSet(mapName=cms.untracked.string('NumberOfCluster'),TopModules=cms.untracked.bool(True),numberTopModules=cms.untracked.uint32(20),RunNumber=cms.untracked.uint32(options.runNumber)),
    cms.PSet(mapName=cms.untracked.string('NumberOfDigi'),TopModules=cms.untracked.bool(True),RunNumber=cms.untracked.uint32(options.runNumber)),
    cms.PSet(mapName=cms.untracked.string('NumberOfOfffTrackCluster'),TopModules=cms.untracked.bool(True),RunNumber=cms.untracked.uint32(options.runNumber)),
    cms.PSet(mapName=cms.untracked.string('NumberOfOfffTrackCluster'),mapSuffix=cms.untracked.string("_autoscale"),mapMax=cms.untracked.double(-1.),RunNumber=cms.untracked.uint32(options.runNumber)),
    cms.PSet(mapName=cms.untracked.string('NumberOfOnTrackCluster'),mapMax=cms.untracked.double(-1.),RunNumber=cms.untracked.uint32(options.runNumber)),
    cms.PSet(mapName=cms.untracked.string('StoNCorrOnTrack'),TopModules=cms.untracked.bool(True),RunNumber=cms.untracked.uint32(options.runNumber),mapMax=cms.untracked.double(35.)), #to be tuned properly
    cms.PSet(mapName=cms.untracked.string('NApvShots'),mapMax=cms.untracked.double(-1.),logScale=cms.untracked.bool(True),RunNumber=cms.untracked.uint32(options.runNumber)),
    cms.PSet(mapName=cms.untracked.string('NApvShots'),mapMax=cms.untracked.double(-1.),logScale=cms.untracked.bool(True),psuMap=cms.untracked.bool(True),loadLVCabling=cms.untracked.bool(True),TopModules=cms.untracked.bool(True),RunNumber=cms.untracked.uint32(options.runNumber)),
#    cms.PSet(mapName=cms.untracked.string('MedianChargeApvShots'),mapMax=cms.untracked.double(-1.)),
#    cms.PSet(mapName=cms.untracked.string('ClusterCharge'),mapMax=cms.untracked.double(-1.)),
#    cms.PSet(mapName=cms.untracked.string('ChargePerCMfromOrigin')),
    cms.PSet(mapName=cms.untracked.string('ChargePerCMfromTrack'),RunNumber=cms.untracked.uint32(options.runNumber),mapMax=cms.untracked.double(-1.)),
    cms.PSet(mapName=cms.untracked.string('NumberMissingHits'),RunNumber=cms.untracked.uint32(options.runNumber),mapMax=cms.untracked.double(-1.)),
    cms.PSet(mapName=cms.untracked.string('NumberValidHits'),RunNumber=cms.untracked.uint32(options.runNumber),mapMax=cms.untracked.double(-1.)),
    cms.PSet(mapName=cms.untracked.string('NumberInactiveHits'),RunNumber=cms.untracked.uint32(options.runNumber))
    )
)

# Services needed for TkHistoMap
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

# Configuration of the SiStripQuality object for the map of bad channels

#process.siStripQualityESProducer.appendToDataLabel = cms.string("test")
process.siStripQualityESProducer.ListOfRecordToMerge=cms.VPSet(
#        cms.PSet( record = cms.string("SiStripDetVOffRcd"),    tag    = cms.string("") ),
#        cms.PSet( record = cms.string("SiStripDetCablingRcd"), tag    = cms.string("") ),
#        cms.PSet( record = cms.string("RunInfoRcd"),           tag    = cms.string("") ),
#        cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
        cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") )
#        cms.PSet( record = cms.string("SiStripBadModuleRcd"),  tag    = cms.string("") )
        )

process.ssqualitystat = cms.EDAnalyzer("SiStripQualityStatistics",
                                       dataLabel = cms.untracked.string(""),
                                       TkMapFileName = cms.untracked.string("PCLBadComponents.png"),  #available filetypes: .pdf .png .jpg .svg
                                       SaveTkHistoMap = cms.untracked.bool(False)
                              )


process.p1 = cms.Path(process.siStripOfflineAnalyser + process.ssqualitystat)
