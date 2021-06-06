import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('sinceRun'
                 , 1
                 , VarParsing.VarParsing.multiplicity.singleton
                 , VarParsing.VarParsing.varType.int
                 , "IOV Start Run Number")
options.register('tag'
                 , 'RPCLBLinkMap_v1'
                 , VarParsing.VarParsing.multiplicity.singleton
                 , VarParsing.VarParsing.varType.string
                 , "Output Data Tag")
options.parseArguments()

process = cms.Process("RPCLBLinkMapPopConAnalyzer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("CondTools.RPC.RPCLBLinkMapPopConAnalyzer_cff")

process.CondDB.connect = 'sqlite_file:data/RPCLinkMap.db'
process.RPCLBLinkMapPopConAnalyzer.Source.dataTag = options.tag;
process.RPCLBLinkMapPopConAnalyzer.Source.txtFile = cms.untracked.string("RPCLBLinkMap.txt");

process.source = cms.Source("EmptyIOVSource"
                            , timetype = cms.string('runnumber')
                            , firstValue = cms.uint64(options.sinceRun)
                            , lastValue = cms.uint64(options.sinceRun)
                            , interval = cms.uint64(1)
)

process.MessageLogger.files.RPCLBLinkMapPopConAnalyzer_log = cms.untracked.PSet(
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
                                                  record = cms.string('RPCLBLinkMapRcd')
                                                  , tag = cms.string(options.tag)
                                              )
                                          )
)

process.p = cms.Path(process.RPCLBLinkMapPopConAnalyzer)
