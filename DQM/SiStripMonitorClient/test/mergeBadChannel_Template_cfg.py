import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("BadChannelMerge")

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
                                    1,
                                    VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                                    VarParsing.VarParsing.varType.int,          # string, int, or float
                                    "run number")
options.parseArguments()


process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout','cerr','MergedBadComponents'), #Reader, cout
    categories = cms.untracked.vstring('SiStripQualityStatistics'),

    debugModules = cms.untracked.vstring('siStripDigis', 
                                         'siStripClusters', 
                                         'siStripZeroSuppression', 
                                         'SiStripClusterizer',
                                         'siStripOfflineAnalyser'),
    cerr = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')
                              ),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO'),
                                default = cms.untracked.PSet(limit=cms.untracked.int32(0))
                              ),
    MergedBadComponents = cms.untracked.PSet(threshold = cms.untracked.string('INFO'),
                                default = cms.untracked.PSet(limit=cms.untracked.int32(0)),
                                SiStripQualityStatistics = cms.untracked.PSet(limit=cms.untracked.int32(100000))
                                )
                                    
)
process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.load("CalibTracker.SiStripESProducers.SiStripBadModuleFedErrESSource_cfi")
process.siStripBadModuleFedErrESSource.appendToDataLabel = cms.string('BadModules_from_FEDBadChannel')
process.siStripBadModuleFedErrESSource.ReadFromFile = cms.bool(True)
process.siStripBadModuleFedErrESSource.FileName = cms.string(options.dqmFile)

process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
       cms.PSet(record = cms.string('SiStripDetCablingRcd'), tag = cms.string('')), # Use Detector cabling information to exclude detectors not connected            
       cms.PSet(record = cms.string('SiStripBadChannelRcd'), tag = cms.string('')), # Online Bad components
       cms.PSet(record = cms.string('RunInfoRcd'), tag = cms.string('')),            # List of FEDs exluded during data taking          
       cms.PSet(record = cms.string('SiStripBadFiberRcd'), tag = cms.string('')),   # Bad Channel list from the selected IOV as done at PCL
       cms.PSet(record = cms.string('SiStripBadModuleFedErrRcd'), tag = cms.string('BadModules_from_FEDBadChannel')) # BadChannel list from FED erroes              
       )
process.siStripQualityESProducer.ReduceGranularity = cms.bool(False)
process.siStripQualityESProducer.ThresholdForReducedGranularity = cms.double(0.3)

process.load("DQMServices.Core.DQMStore_cfg")

process.stat = cms.EDAnalyzer("SiStripQualityStatistics",
    TkMapFileName = cms.untracked.string('MergedBadComponentsTkMap.png'),
    dataLabel = cms.untracked.string('')
)
#### Add these lines to produce a tracker map
process.load("DQMServices.Core.DQMStore_cfg")
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")


process.p = cms.Path(process.stat)

