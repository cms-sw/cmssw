import FWCore.ParameterSet.Config as cms

from FWCore.PythonUtilities.LumiList import LumiList
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")
options.register("runList"
                 , [320855]
                 , VarParsing.multiplicity.list
                 , VarParsing.varType.int
                 , "Run selection")
options.register("lumiList"
                 , "step1_lumiRanges.log"
                 , VarParsing.multiplicity.singleton
                 , VarParsing.varType.string
                 , "JSON file")
options.parseArguments()
 

process = cms.Process("testRPCNewReadoutDQM")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("FWCore.MessageLogger.MessageLogger_cfi")


# process.MessageLogger.cerr.FwkReport.reportEvery = 10

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.GlobalTag.globaltag = "80X_dataRun2_Express_v15"
# process.GlobalTag.globaltag = "101X_dataRun2_Prompt_v10"
process.GlobalTag.globaltag = "102X_dataRun2_Prompt_v1"



#######################################################
### RPC RawToDigi
### RPC RawToDigi - from Legacy
process.load("EventFilter.RPCRawToDigi.rpcUnpackingModule_cfi")

### RPC RawToDigi - from TwinMux
process.load("EventFilter.RPCRawToDigi.RPCTwinMuxRawToDigi_cff")

### RPC RawToDigi - from CPPF
process.load("EventFilter.RPCRawToDigi.RPCCPPFRawToDigi_cff")
# process.load("EventFilter.RPCRawToDigi.RPCCPPFRawToDigi_sqlite_cff")

### RPC RawToDigi - from OMTF
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.omtfStage2Digis = cms.EDProducer("OmtfUnpacker",
  inputLabel = cms.InputTag('rawDataCollector'),
)

### RPC Digi Merger
process.load("EventFilter.RPCRawToDigi.RPCDigiMerger_cff")


#######################################################
### RPCRecHit - from legacy
process.load('RecoLocalMuon.RPCRecHit.rpcRecHits_cfi')
from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import *
process.rpcLegacyRecHits = process.rpcRecHits.clone()
process.rpcLegacyRecHits.rpcDigiLabel = cms.InputTag('rpcUnpackingModule')
### RPCRecHit - from TwinMux
process.rpcTwinMuxRecHits = process.rpcRecHits.clone()
process.rpcTwinMuxRecHits.rpcDigiLabel = cms.InputTag('RPCTwinMuxRawToDigi')
### RPCRecHit - from CPPF
process.rpcCPPFRecHits = process.rpcRecHits.clone()
process.rpcCPPFRecHits.rpcDigiLabel = cms.InputTag('rpcCPPFRawToDigi')
### RPCRecHit - from OMTF
process.rpcOMTFRecHits = process.rpcRecHits.clone()
process.rpcOMTFRecHits.rpcDigiLabel = cms.InputTag('omtfStage2Digis')
### RPCRecHit - from Merger
process.rpcMergerRecHits = process.rpcRecHits.clone()
process.rpcMergerRecHits.rpcDigiLabel = cms.InputTag('RPCDigiMerger')



### Print Summary
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )


#######################################################
### DQM - from legacy
process.load("DQM.RPCMonitorDigi.RPCDigiMonitoring_cfi")
process.rpcdigidqm.UseRollInfo = True
process.rpcdigidqm.UseMuon =  False
process.rpcdigidqm.NoiseFolder = cms.untracked.string("AllHitsLegacy")
process.rpcdigidqm.RecHitLabel = cms.InputTag("rpcLegacyRecHits")
### DQM - from TwinMux
process.rpcTwinMuxdigidqm = process.rpcdigidqm.clone()
process.rpcTwinMuxdigidqm.NoiseFolder = cms.untracked.string("AllHitsTwinMux")
process.rpcTwinMuxdigidqm.RecHitLabel = cms.InputTag("rpcTwinMuxRecHits")
### DQM - from CPPF
process.rpcCPPFdigidqm = process.rpcdigidqm.clone()
process.rpcCPPFdigidqm.NoiseFolder = cms.untracked.string("AllHitsCPPF")
process.rpcCPPFdigidqm.RecHitLabel = cms.InputTag("rpcCPPFRecHits")
### DQM - from OMTF
process.rpcOMTFdigidqm = process.rpcdigidqm.clone()
process.rpcOMTFdigidqm.NoiseFolder = cms.untracked.string("AllHitsOMTF")
process.rpcOMTFdigidqm.RecHitLabel = cms.InputTag("rpcOMTFRecHits")
### DQM - from Merger
process.rpcMergerdigidqm = process.rpcdigidqm.clone()
process.rpcMergerdigidqm.NoiseFolder = cms.untracked.string("AllHitsMerger")
process.rpcMergerdigidqm.RecHitLabel = cms.InputTag("rpcMergerRecHits")



#######################################################
### DQM Saver
process.load("Configuration.StandardSequences.DQMSaverAtJobEnd_cff")
process.dqmEnv.subSystemFolder = 'RPC'




# Source
process.source = cms.Source("PoolSource"
                            , fileNames = cms.untracked.vstring(options.inputFiles)
                            # , fileNames = cms.untracked.vstring("/store/data/Run2018D/SingleMuon/RAW/v1/000/320/855/00000/1822B685-AC98-E811-9A30-FA163E4D9C0E.root")
                            # , fileNames = cms.untracked.vstring("file:/eos/cms/store/data/Run2018D/SingleMuon/RAW/v1/000/320/500/00000/BC8C0D64-EF93-E811-81A5-02163E00C22E.root")
                            # , lumisToProcess = lumilist.getVLuminosityBlockRange()
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(3000) )
# process.maxLuminosityBlocks = cms.untracked.PSet(input = cms.untracked.int32(10))



process.p = cms.Path( 
                      (process.rpcUnpackingModule + process.RPCTwinMuxRawToDigi + process.rpcCPPFRawToDigi + process.omtfStage2Digis) 
                      * process.RPCDigiMerger 
                      * (process.rpcLegacyRecHits + process.rpcTwinMuxRecHits + process.rpcCPPFRecHits + process.rpcOMTFRecHits + process.rpcMergerRecHits)
                      * (process.rpcdigidqm + process.rpcTwinMuxdigidqm + process.rpcCPPFdigidqm + process.rpcOMTFdigidqm + process.rpcMergerdigidqm)
                      * process.DQMSaver
                    )
# process.p = cms.Path(process.RPCTwinMuxRawToDigi )

# Output
process.out = cms.OutputModule("PoolOutputModule"
                               , outputCommands = cms.untracked.vstring("drop *"
                                                                        , "keep *_*_*_testRPCTwinMux"
                                                                        , "keep *_rpcRecHits_*_*")
                               # , fileName = cms.untracked.string(options.outputFile)
                               , fileName = cms.untracked.string("testRPCNewReadoutDQM.root")
                               , SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("p"))
)

# process.e = cms.EndPath(process.out)