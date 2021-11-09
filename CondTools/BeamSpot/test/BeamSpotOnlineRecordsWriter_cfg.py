import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("write2DB")

options = VarParsing.VarParsing()
options.register('unitTest',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "are we running the unit test?")
options.register('inputFile',
                 "BeamFitResults_Run306171.txt", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "location of the input data")
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


process.load("FWCore.MessageLogger.MessageLogger_cfi")
from CondCore.CondDB.CondDB_cfi import *

if options.unitTest :
    if options.inputRecord ==  "BeamSpotOnlineLegacyObjectsRcd" : 
        tag_name = 'BSLegacy_tag'
    else:
        tag_name = 'BSHLT_tag'
else:
    tag_name = options.inputTag

#################################
# Produce a SQLITE FILE
#################################
CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('sqlite_file:test_%s.db' % tag_name)) # choose an output name
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBBeamSpotObjects,
                                          timetype = cms.untracked.string('lumiid'), #('lumiid'), #('runnumber')
                                          toPut = cms.VPSet(cms.PSet(record = cms.string(options.inputRecord), # BeamSpotOnline record
                                                                     tag = cms.string(tag_name))),             # choose your favourite tag
                                          loadBlobStreamer = cms.untracked.bool(False)
                                          )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.beamspotonlinewriter = cms.EDAnalyzer("BeamSpotOnlineRecordsWriter",
                                              isHLT = cms.bool((options.inputRecord ==  "BeamSpotOnlineHLTObjectsRcd")),
                                              InputFileName = cms.untracked.string(options.inputFile), # choose your input file
                                              )

if(options.startRun>0 and options.startLumi>0):
    process.beamspotonlinewriter.IOVStartRun = cms.untracked.uint32(options.startRun)    # Customize your Run
    process.beamspotonlinewriter.IOVStartLumi = cms.untracked.uint32(options.startLumi)  # Customize your Lumi


process.p = cms.Path(process.beamspotonlinewriter)
