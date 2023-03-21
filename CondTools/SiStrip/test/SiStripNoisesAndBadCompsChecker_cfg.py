from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import copy 

process = cms.Process("Demo")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "auto:phase1_2018_cosmics_peak",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "GlobalTag")

options.register ('runNumber',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,            # string, int, or float
                  "run number")

options.register ('writePayload',
                  True,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,            # string, int, or float
                  "write out payload")

options.parseArguments()


##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripNoisesAndBadCompsChecker=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripNoisesAndBadCompsChecker = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

##
## Conditions
##
process.load("Configuration.Geometry.GeometryRecoDB_cff") # Ideal geometry and interface 
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

print("Using Global Tag:", process.GlobalTag.globaltag._value)

##
## Empty Source
##
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.runNumber),
                            numberEventsInRun = cms.untracked.uint32(1),
                            )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

##
## DB Output
##
if(options.writePayload) :
    process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                              BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                              DBParameters = cms.PSet(authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')),
                                              timetype = cms.untracked.string('runnumber'),
                                              connect = cms.string('sqlite_file:SiStripNoise_PeakMode_2018_Minus20C_v0_mc_fixed.db'),
                                              toPut = cms.VPSet(cms.PSet(record = cms.string('SiStripNoisesRcd'),
                                                                         tag = cms.string('SiStripNoise_PeakMode_2018_Minus20C_v0_mc_fixed')
                                                                     )
                                                            )
                                          )

##
## Analyzer
##
process.demo = cms.EDAnalyzer('SiStripNoisesAndBadCompsChecker',
                              writePayload = cms.untracked.bool(options.writePayload),
                              printDebug = cms.untracked.uint32(100),
                              file = cms.untracked.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'))

##
## Parh
##
process.p = cms.Path(process.demo)
