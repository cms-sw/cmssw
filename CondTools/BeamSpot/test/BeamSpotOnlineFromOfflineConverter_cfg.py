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
process.MessageLogger.cerr.FwkReport.reportEvery = 100000 # do not clog output with IO

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

####################################################################
# Empty source 
####################################################################
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.startRun),                  
                            firstLuminosityBlock = cms.untracked.uint32(options.startLumi),     
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),            
                            numberEventsInRun = cms.untracked.uint32(1))

####################################################################
# Connect to conditions DB
####################################################################

# either from Global Tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,"auto:phase2_realistic")

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

####################################################################
# Load and configure outputservice
####################################################################
if options.unitTest :
    if options.inputRecord ==  "BeamSpotOnlineLegacyObjectsRcd" : 
        tag_name = 'BSLegacy_tag'
    else:
        tag_name = 'BSHLT_tag'
else:
    tag_name = options.inputTag

from CondCore.CondDB.CondDB_cfi import *
CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('sqlite_file:test_%s.db' % tag_name)) # choose an output name
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBBeamSpotObjects,
                                          timetype = cms.untracked.string('lumiid'), #('lumiid'), #('runnumber')
                                          toPut = cms.VPSet(cms.PSet(record = cms.string(options.inputRecord), # BeamSpotOnline record
                                                                     tag = cms.string(tag_name))),             # choose your favourite tag
                                          loadBlobStreamer = cms.untracked.bool(False)
                                          )

isForHLT = (options.inputRecord == "BeamSpotOnlineHLTObjectsRcd")
print("isForHLT: ",isForHLT)

####################################################################
# Load and configure analyzer
####################################################################
from CondTools.BeamSpot.beamSpotOnlineFromOfflineConverter_cfi import beamSpotOnlineFromOfflineConverter
process.BeamSpotOnlineFromOfflineConverter = beamSpotOnlineFromOfflineConverter.clone(isHLT = isForHLT)

# Put module in path:
process.p = cms.Path(process.BeamSpotOnlineFromOfflineConverter)
