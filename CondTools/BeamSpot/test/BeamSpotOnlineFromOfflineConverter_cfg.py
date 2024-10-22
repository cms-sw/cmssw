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
                 "myInputTagName", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "input tag name")
options.register('inputFile',
                 "", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "input sqlite (.db) file")
options.register('outputTag',
                 "myOutputTagName", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "output tag name")
options.register('outputRecord',
                 "BeamSpotOnlineLegacyObjectsRcd", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "type of output record")
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

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(999))

####################################################################
# Empty source 
####################################################################
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.startRun),                  
                            firstLuminosityBlock = cms.untracked.uint32(options.startLumi),     
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),            
                            numberEventsInRun = cms.untracked.uint32(999))

####################################################################
# Connect to conditions DB
####################################################################
if options.inputFile != "":
    from CondCore.CondDB.CondDB_cfi import *
    # Load from a local db file
    CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('sqlite_file:'+options.inputFile))
    #To connect directly to a database replace the above line with:
    #CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))

    # Specifying the tag name:
    process.dbInput = cms.ESSource("PoolDBESSource",
                                   CondDBBeamSpotObjects,
                                   toGet = cms.VPSet(cms.PSet(record = cms.string('BeamSpotObjectsRcd'),
                                                              tag = cms.string(options.inputTag) # choose your input tag
                                                             )
                                                    )
                                  )
else:
    # Load from Global Tag
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag,"auto:run2_data")

####################################################################
# Load and configure outputservice
####################################################################
if options.unitTest :
    if options.outputRecord ==  "BeamSpotOnlineLegacyObjectsRcd" :
        tag_name = 'BSLegacy_tag'
    else:
        tag_name = 'BSHLT_tag'
else:
    tag_name = options.outputTag

from CondCore.CondDB.CondDB_cfi import *
CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('sqlite_file:test_%s.db' % tag_name)) # choose an output name
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBBeamSpotObjects,
                                          timetype = cms.untracked.string('lumiid'), #('lumiid'), #('runnumber')
                                          toPut = cms.VPSet(cms.PSet(record = cms.string(options.outputRecord), # BeamSpotOnline record
                                                                     tag = cms.string(tag_name))),              # choose your favourite tag
                                          loadBlobStreamer = cms.untracked.bool(False)
                                          )

isForHLT = (options.outputRecord == "BeamSpotOnlineHLTObjectsRcd")
print("isForHLT: ",isForHLT)

####################################################################
# Load and configure analyzer
####################################################################
from CondTools.BeamSpot.beamSpotOnlineFromOfflineConverter_cfi import beamSpotOnlineFromOfflineConverter
process.BeamSpotOnlineFromOfflineConverter = beamSpotOnlineFromOfflineConverter.clone(isHLT = isForHLT)
process.BeamSpotOnlineFromOfflineConverter.IOVStartRun  = options.startRun
process.BeamSpotOnlineFromOfflineConverter.IOVStartLumi = options.startLumi

# Put module in path:
process.p = cms.Path(process.BeamSpotOnlineFromOfflineConverter)
