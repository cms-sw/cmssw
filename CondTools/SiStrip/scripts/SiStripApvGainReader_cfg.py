#! /bin/env cmsRun
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


options = VarParsing('python')
options.register('runN', 1,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "runN in the IOV"
)

options.parseArguments()

if (not options.tag or not options.inputFiles):
    raise ValueError('usage: cmsRun SiStripApvGainReader_cfg.py inputFiles=my_input_file  tag=my_tag')
process = cms.Process("SiStripApvGainReader")

process.MessageLogger = cms.Service(
    "MessageLogger",
    debugModules = cms.untracked.vstring(''),
    threshold = cms.untracked.string('INFO'),
    destinations = cms.untracked.vstring('SiStripApvGainReader.log')
    )

##
## Empty events source
## (specify the run number to pick correct IOV) 
##
process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(1),
                            firstRun = cms.untracked.uint32(options.runN)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

##
## To Correctly build the SiStripGainRcd in the ES
## we need all the dependent records => specify a Global Tag  
##
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')


##
## Prefer the SiStripApvGainRcd or SiStripApvGain2Rcd
## under test  
##
process.poolDBESSource = cms.ESSource("PoolDBESSource",
                                      #connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'), 
                                      connect = cms.string(options.inputFiles[0]),
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                                                 tag = cms.string(options.tag)
                                                                 )
                                                        )
                                      )
process.prefer_poolDBESSource = cms.ESPrefer("PoolDBESSource","poolDBESSource")

##
## Dump the Gain payload. It will be in the format
## DetId APV1 APV2 APV3 APV4 (APV5) (APV6) 
##
process.gainreader = cms.EDAnalyzer("SiStripApvGainReader",
                                    printDebug = cms.untracked.bool(True),
                                    outputFile = cms.untracked.string("SiStripApvGains_dump.txt"),
                                    gainType   = cms.untracked.uint32(0)    #0 for G1, 1 for G2
                                    )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("gain.root"),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.p1 = cms.Path(process.gainreader)


