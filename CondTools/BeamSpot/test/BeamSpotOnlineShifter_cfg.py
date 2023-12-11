import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("READ")

options = VarParsing.VarParsing()
options.register('inputTag',
                 "myTagName", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "output tag name")
options.register('inputRecord',
                 "BeamSpotOnlineLegacyObjectsRcd", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "type of record")
options.register('startRun',
                 306171, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.int, # string, int, or float
                 "location of the input data")
options.register('startLumi',
                 497, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.int, # string, int, or float
                 "IOV Start Lumi")
options.register('maxLSToRead',
                 10, ## default value for unit test
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.int, # string, int, or float
                 "total number of LumiSections to read in input")
options.parseArguments()

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100000 # do not clog output with IO

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxLSToRead))   # large number of events is needed since we probe 5000LS for run (see below)

####################################################################
# Empty source 
####################################################################
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.startRun),                  # Run in ../data/BeamFitResults_Run306171.txt
                            firstLuminosityBlock = cms.untracked.uint32(options.startLumi),         # Lumi in ../data/BeamFitResults_Run306171.txt
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),  # probe one event per LS
                            numberEventsInRun = cms.untracked.uint32(5000),           # a number of events > the number of LS possible in a real run (5000 s ~ 32 h)
                            )

####################################################################
# Connect to conditions DB
####################################################################
process.load("Configuration.StandardSequences.GeometryDB_cff") # for the topolgy
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "132X_dataRun3_HLT_v2")
process.GlobalTag.toGet =  cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),             # record
                                              tag = cms.string("TrackerAlignment_PCL_byRun_v0_hlt"),  # choose your favourite tag
                                              label = cms.untracked.string("reference")),             # refence label
                                     cms.PSet(record = cms.string("TrackerAlignmentRcd"),                  # record
                                              tag = cms.string("TrackerAlignment_collisions23_forHLT_v9"), # choose your favourite tag
                                              label = cms.untracked.string("target")))                     # target label

#process.GlobalTag.DumpStat = cms.untracked.bool(True)

myTagName = options.inputTag

print("isForHLT: ",(options.inputRecord ==  "BeamSpotOnlineHLTObjectsRcd"))
print("max LS to Read: ",options.maxLSToRead)

#################################
# Produce a SQLITE FILE
#################################
from CondCore.CondDB.CondDB_cfi import *
CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('sqlite_file:test_%s.db' % myTagName)) # choose an output name
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBBeamSpotObjects,
                                          timetype = cms.untracked.string('lumiid'), #('lumiid'), #('runnumber')
                                          toPut = cms.VPSet(cms.PSet(record = cms.string(options.inputRecord), # BeamSpotOnline record
                                                                     tag = cms.string(myTagName))),             # choose your favourite tag
                                          loadBlobStreamer = cms.untracked.bool(False)
                                          )

####################################################################
# Load and configure analyzer
####################################################################
process.beamspotonlineshifter = cms.EDAnalyzer("BeamSpotOnlineShifter",
                                               isHLT = cms.bool((options.inputRecord ==  "BeamSpotOnlineHLTObjectsRcd")),
                                               xShift =  cms.double(+0.000141),
                                               yShift =  cms.double(+0.000826),
                                               zShift =  cms.double(+0.000277))
                                   
# Put module in path:
process.p = cms.Path(process.beamspotonlineshifter)
