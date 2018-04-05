import FWCore.ParameterSet.Config as cms

from FWCore.PythonUtilities.LumiList import LumiList
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")
options.register("runList"
                 , []
                 , VarParsing.multiplicity.list
                 , VarParsing.varType.int
                 , "Run selection")
options.register("lumiList"
                 , "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/DCSOnly/json_DCSONLY.txt"
                 , VarParsing.multiplicity.singleton
                 , VarParsing.varType.string
                 , "JSON file")
options.parseArguments()

lumilist = LumiList(filename = options.lumiList)
if len(options.runList) :
    runlist = LumiList(runs = options.runList)
    lumilist = lumilist & runlist
    if not len(lumilist) :
        raise RuntimeError("The resulting LumiList is empty")

process = cms.Process("testRPCTwinMux")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "80X_dataRun2_Express_v15"

process.load("EventFilter.RPCRawToDigi.RPCTwinMuxRawToDigi_sqlite_cff")
process.load("EventFilter.RPCRawToDigi.RPCTwinMuxDigiToRaw_sqlite_cff")
process.RPCTwinMuxDigiToRaw.inputTag = cms.InputTag("RPCTwinMuxRawToDigi")

process.load("EventFilter.RPCRawToDigi.rpcPacker_cfi")
process.rpcpacker.InputLabel = cms.InputTag("RPCTwinMuxRawToDigi")
process.load("EventFilter.RPCRawToDigi.rpcUnpackingModule_cfi")
process.rpcUnpackingModulePacked = process.rpcUnpackingModule.clone()
process.rpcUnpackingModulePacked.InputLabel = cms.InputTag("rpcpacker")

process.RPCTwinMuxDigiToRawPAC = process.RPCTwinMuxDigiToRaw.clone()
process.RPCTwinMuxDigiToRawPAC.inputTag = cms.InputTag("rpcUnpackingModule")
process.RPCTwinMuxRawToDigiPAC = process.RPCTwinMuxRawToDigi.clone()
process.RPCTwinMuxRawToDigiPAC.inputTag = cms.InputTag("RPCTwinMuxDigiToRawPAC")

process.RPCTwinMuxRawToDigiPacked = process.RPCTwinMuxRawToDigi.clone()
process.RPCTwinMuxRawToDigiPacked.inputTag = cms.InputTag("RPCTwinMuxDigiToRaw")

process.load("EventFilter.L1TXRawToDigi.twinMuxStage2Digis_cfi")
process.twinMuxStage2DigisPacked = process.twinMuxStage2Digis.clone()
process.twinMuxStage2DigisPacked.DTTM7_FED_Source = cms.InputTag("RPCTwinMuxDigiToRaw")

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Source
process.source = cms.Source("PoolSource"
                            , fileNames = cms.untracked.vstring(options.inputFiles)
                            , lumisToProcess = lumilist.getVLuminosityBlockRange()
)

#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
process.maxLuminosityBlocks = cms.untracked.PSet(input = cms.untracked.int32(10))

process.p = cms.Path( ( process.rpcUnpackingModule * process.RPCTwinMuxDigiToRawPAC * process.RPCTwinMuxRawToDigiPAC )
                      + ( process.RPCTwinMuxRawToDigi
                          * ( ( process.rpcpacker * process.rpcUnpackingModulePacked )
                              + ( process.RPCTwinMuxDigiToRaw
                                  * ( process.RPCTwinMuxRawToDigiPacked + process.twinMuxStage2DigisPacked ) )
                          )
                      )
                      + process.twinMuxStage2Digis
)

# Output
process.out = cms.OutputModule("PoolOutputModule"
                               , outputCommands = cms.untracked.vstring("drop *"
                                                                        , "keep *_*_*_testRPCTwinMux")
                               , fileName = cms.untracked.string(options.outputFile)
                               #, fileName = cms.untracked.string("testRPCTwinMux.root")
                               , SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("p"))
)

process.e = cms.EndPath(process.out)
