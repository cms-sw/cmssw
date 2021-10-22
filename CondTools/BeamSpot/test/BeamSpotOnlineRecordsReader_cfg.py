import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("READ")

options = VarParsing.VarParsing()
options.register('unitTest',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "are we running the unit test?")
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
options.parseArguments()

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100000                         # do not clog output with IO

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1 if options.unitTest else 10000000) )   # large number of events is needed since we probe 5000LS for run (see below)

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

# either from Global Tag
# process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
# from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag,"auto:run2_data")

# ...or specify database connection and tag:  
#from CondCore.CondDB.CondDB_cfi import *
#CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))
#process.dbInput = cms.ESSource("PoolDBESSource",
#                               CondDBBeamSpotObjects,
#                               toGet = cms.VPSet(cms.PSet(record = cms.string('BeamSpotOnlineLegacyObjectsRcd'), # BeamSpotOnlineLegacy record
#                                                          tag = cms.string('BSLegacy_tag')                       # choose your favourite tag
#                                                          )
#                                                 )
#                               )
# ...or from a local db file
# input database (in this case the local sqlite file)

if options.unitTest :
    if options.inputRecord ==  "BeamSpotOnlineLegacyObjectsRcd" : 
        tag_name = 'BSLegacy_tag'
    else:
        tag_name = 'BSHLT_tag'
else:
    tag_name = options.inputTag

from CondCore.CondDB.CondDB_cfi import *
CondDBBeamSpotOnlineLegacy = CondDB.clone(connect = cms.string("sqlite_file:test_%s.db" % tag_name)) # customize with input db file
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBBeamSpotOnlineLegacy,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string(options.inputRecord),  # BeamSpotOnline record
        tag = cms.string(tag_name)                 # choose your favourite tag
    ))
)

print("isForHLT: ",(options.inputRecord ==  "BeamSpotOnlineHLTObjectsRcd"))

####################################################################
# Load and configure analyzer
####################################################################
process.beamspotonlinereader = cms.EDAnalyzer("BeamSpotOnlineRecordsReader",
                                              isHLT = cms.bool((options.inputRecord ==  "BeamSpotOnlineHLTObjectsRcd")),
                                              rawFileName = cms.untracked.string("test.txt") # choose an output name
                                              )

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("test.root") # choose an output name
                                   )

# Put module in path:
process.p = cms.Path(process.beamspotonlinereader)
