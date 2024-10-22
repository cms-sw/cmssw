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
options.register('startRun',
                 1, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.int, # string, int, or float
                 "location of the input data")
options.register('startLumi',
                 1, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.int, # string, int, or float
                 "IOV Start Lumi")
options.parseArguments()

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000000                 # do not clog output with IO

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )   # large number of events is needed since we probe 5000LS for run (see below)

####################################################################
# Empty source 
####################################################################

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.startRun),
                            firstLuminosityBlock = cms.untracked.uint32(options.startRun),  # probe one LS after the other
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),        # probe one event per LS
                            numberEventsInRun = cms.untracked.uint32(1),                    # a number of events > the number of LS possible in a real run (5000 s ~ 32 h)
                            )

####################################################################
# Connect to conditions DB
####################################################################

if options.unitTest:
    tag_name = 'simBS_tag'
else:
    tag_name = options.inputTag

# either from Global Tag
# process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
# from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag,"auto:phase1_2023_realistic")

# ...or specify database connection and tag...
# from CondCore.CondDB.CondDB_cfi import *
# CondDBSimBeamSpot = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))
# process.dbInput = cms.ESSource("PoolDBESSource",
#                                CondDBSimBeamSpot,
#                                toGet = cms.VPSet(cms.PSet(record = cms.string('SimBeamSpotObjectsRcd'),
#                                                           tag = cms.string(tag_name)  # customize with input tag name
#                                                           )
#                                                  )
#                                )

# ...or specify local db file:
from CondCore.CondDB.CondDB_cfi import *
CondDBSimBeamSpot = CondDB.clone(connect = cms.string("sqlite_file:test_%s.db" % tag_name)) # customize with input db file
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBSimBeamSpot,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SimBeamSpotObjectsRcd'),
        tag = cms.string(tag_name)  # customize with input tag name
    ))
)

####################################################################
# Load and configure analyzer
####################################################################
from CondTools.BeamSpot.beamProfile2DBReader_cfi import beamProfile2DBReader
process.BeamProfile2DBRead = beamProfile2DBReader.clone(rawFileName = 'reference_SimBeamSpotObjects.txt')

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("reference_SimBeamSpotObjects.root")
                                   ) 

# Put module in path:
process.p = cms.Path(process.BeamProfile2DBRead)
