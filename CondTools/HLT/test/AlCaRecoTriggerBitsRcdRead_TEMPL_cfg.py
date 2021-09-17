import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing 

process = cms.Process("READ")

options = VarParsing.VarParsing()
options.register( "inputDB", 
                  "frontier://FrontierProd/CMS_CONDITIONS",  #default value
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.string,
                  "the input DB"
                  )

options.register( "inputTag", 
                  "AlCaRecoHLTpaths8e29_1e31_v7_hlt",  #default value
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.string,
                  "the input tag"
                  )

options.parseArguments()


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1000)
    ))

# the module writing to DB
process.load("CondTools.HLT.AlCaRecoTriggerBitsRcdRead_cfi")
# 'twiki' is default - others are text, python (future: html?)
#process.AlCaRecoTriggerBitsRcdRead.outputType = 'twiki'
# If rawFileName stays empty (default), use the message logger for output.
# Otherwise use the file name specified, adding a suffix according to outputType:
process.AlCaRecoTriggerBitsRcdRead.rawFileName = 'triggerBits'+options.inputTag

# No data, but might want to specify the 'firstRun' to check (default is 1):
process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(1), # do not change!
                            firstRun = cms.untracked.uint32(1)
                            )
# With 'numberEventsInRun = 1' above,
# this will check IOVs until run (!) number specified as 'input' here,
# so take care to choose a one that is not too small...:
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500000) )

# Input for AlCaRecoTriggerBitsRcd,
# either via GlobalTag
# (loading of Configuration.StandardSequences.CondDBESSource_cff equivalent to CondCore.ESSources.CondDBESSource_cfi
# as entry point for condition records in the EventSetup,
# but sufficient and faster than Configuration.StandardSequences.FrontierConditions_GlobalTag_cff):
#from Configuration.AlCa.autoCond import autoCond
#process.load("Configuration.StandardSequences.CondDBESSource_cff")
#process.GlobalTag.globaltag = autoCond['run2_data'] #choose your tag

# ...or specify database and tag:  
from CondCore.CondDB.CondDB_cfi import *
#CondDBTriggerBits = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_COND_31X_HLT'))
CondDBTriggerBits = CondDB.clone(connect = cms.string(options.inputDB))
process.dbInput = cms.ESSource("PoolDBESSource",
                              CondDBTriggerBits,
                              toGet = cms.VPSet(cms.PSet(record = cms.string('AlCaRecoTriggerBitsRcd'),
                                                         tag = cms.string(options.inputTag) # choose tag you want
                                                         )
                                                )
                              )

# Put module in path:
process.p = cms.Path(process.AlCaRecoTriggerBitsRcdRead)
