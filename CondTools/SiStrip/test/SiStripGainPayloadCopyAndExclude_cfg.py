'''
This file is an example configuration of the SiStripPayloadCopyAndExclude module.
This module is meant to copy the content of a SiStrip APV Gain payload (either G1 or G2) 
from a local sqlite file (that should be feeded to the Event Setup via the SiStripApvGain3Rcd  
and put in another local sqlite file, excepted for the modules specified in the excludedModules 
parameter. If the doReverse parameter is true, the opposite action is performed. 
'''

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("SiStripPayloadCopyAndExclude")

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "101X_dataRun2_Express_v7",
                  VarParsing.VarParsing.multiplicity.singleton,  # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.register ('runNumber',
                  317478,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,            # string, int, or float
                  "run number")

options.register ('doReverse',
                  False,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,           # string, int, or float
                  "reverse the selection")

options.parseArguments()


if(options.doReverse): 
    print("====================================================================================================================================")
    print("%MSG-i DoReverse: : Going to revert the selection. All modules will be taken from GT, unless they are specified in the modules list!")
    print("====================================================================================================================================")

##
## Messages
##
process.load("FWCore.MessageService.MessageLogger_cfi")

##
## Event Source
##
process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(options.runNumber),
                            lastValue  = cms.uint64(options.runNumber),
                            timetype  = cms.string('runnumber'),
                            interval = cms.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

##
## Conditions inputs
##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')
process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("SiStripApvGain3Rcd"),
             tag = cms.string("SiStripApvGainAAG_pcl"),
             #connect = cms.string("sqlite_file:/eos/cms/store/group/alca_global/multiruns/results/prod//slc6_amd64_gcc630/CMSSW_10_1_5/86791_1p_0f/promptCalibConditions86791.db")             
             connect = cms.string("sqlite_file:promptCalibConditions86791.db") # locally copied file for unit test
             )
    )

##
## Worker module
##
process.SiStripGainPayloadCopyAndExclude = cms.EDAnalyzer('SiStripGainPayloadCopyAndExclude',
                                                          ### FED 387
                                                          excludedModules = cms.untracked.vuint32(436281608,436281604,436281592,436281624,436281620,436281644,436281640,436281648,436281668,436281680,436281684,436281688,436281720,436281700,436281708,436281556,436281552,436281704,436281764,436281768,436281572,436281576,436281748,436281744,436281740,436281780,436281784,436281612,436281616,436281588,436281580,436281584,436281636,436281656,436281652,436281676,436281672,436281732,436281736,436281716,436281712,436281776,436281772,436281548,436281544,436281540,436281752,436281560),
                                                          reverseSelection = cms.untracked.bool(options.doReverse),  # if True it will take everything from GT, but the execludedModules from the Gain3 tag 
                                                          record   = cms.untracked.string("SiStripApvGainRcd"),
                                                          gainType = cms.untracked.uint32(1) # 0 for G1, 1 for G2
)

##
## Output database (in this case local sqlite file)
##
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = "sqlite_file:modifiedGains_"+process.GlobalTag.globaltag._value+'_IOV_'+str(options.runNumber)+("_reverse.db" if options.doReverse else ".db")
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                                                     tag = cms.string('modifiedGains')
                                                                     )
                                                            )
)

process.p = cms.Path(process.SiStripGainPayloadCopyAndExclude)
