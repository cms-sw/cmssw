import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('sinceRun'
                 , 1
                 , VarParsing.VarParsing.multiplicity.singleton
                 , VarParsing.VarParsing.varType.int
                 , "IOV Start Run Number")
options.register('tag'
                 , 'RPCOMTFLinkMap_v1'
                 , VarParsing.VarParsing.multiplicity.singleton
                 , VarParsing.VarParsing.varType.string
                 , "Output Data Tag")
options.parseArguments()

process = cms.Process("RPCOMTFLinkMapPopConAnalyzer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("CondTools.RPC.RPCOMTFLinkMapPopConAnalyzer_cff")

process.CondDB.connect = 'sqlite_file:data/RPCLinkMap.db'
process.RPCOMTFLinkMapPopConAnalyzer.Source.dataTag = options.tag;
process.RPCOMTFLinkMapPopConAnalyzer.Source.txtFile = cms.untracked.string("RPCOMTFLinkMap.txt");
process.RPCOMTFLinkMapPopConAnalyzer.Source.sinceRun = cms.uint64(options.sinceRun)

process.source = cms.Source("EmptyIOVSource"
                            , timetype = cms.string('runnumber')
                            , firstValue = cms.uint64(options.sinceRun)
                            , lastValue = cms.uint64(options.sinceRun)
                            , interval = cms.uint64(1)
)

process.MessageLogger.files.RPCOMTFLinkMapPopConAnalyzer_log = cms.untracked.PSet(
    threshold = cms.untracked.string("INFO")
    , FwkReport = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(1)
    )
)

process.MessageLogger.cout.threshold = cms.untracked.string("INFO")

process.PoolDBOutputService = cms.Service("PoolDBOutputService"
                                          , process.CondDB
                                          , timetype = cms.untracked.string('runnumber')
                                          , toPut = cms.VPSet(
                                              cms.PSet(
                                                  record = cms.string('RPCOMTFLinkMapRcd')
                                                  , tag = cms.string(options.tag)
                                              )
                                          )
)

process.p = cms.Path(process.RPCOMTFLinkMapPopConAnalyzer)
