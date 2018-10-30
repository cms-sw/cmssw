import FWCore.ParameterSet.Config as cms

from FWCore.PythonUtilities.LumiList import LumiList
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")

process = cms.Process("testRPCDigiMerger")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "102X_dataRun2_Sep2018Rereco_test_v1"

#######################################################
### RPC RawToDigi
### RPC RawToDigi - from Legacy
process.load("EventFilter.RPCRawToDigi.rpcUnpackingModule_cfi")

### RPC RawToDigi - from TwinMux
process.load("EventFilter.RPCRawToDigi.RPCTwinMuxRawToDigi_cff")

### RPC RawToDigi - from CPPF
process.load("EventFilter.RPCRawToDigi.RPCCPPFRawToDigi_cff")
# process.load("EventFilter.RPCRawToDigi.RPCCPPFRawToDigi_sqlite_cff") #to load CPPF link maps from the local DB

### RPC RawToDigi - from OMTF
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.omtfStage2Digis = cms.EDProducer("OmtfUnpacker",
  inputLabel = cms.InputTag('rawDataCollector'),
)

process.load("EventFilter.RPCRawToDigi.RPCDigiMerger_cff")

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Source
process.source = cms.Source("PoolSource"
                            , fileNames = cms.untracked.vstring(
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/34C0D29C-E3AA-E811-84FB-FA163EF5DB03.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/DEF27B09-E7AA-E811-B671-FA163EDF3211.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/163BB720-E7AA-E811-988D-FA163EC62C4D.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/B06CF0A5-E9AA-E811-AC1F-FA163EF7BA8C.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/AADE09C6-E9AA-E811-97DC-FA163E516F48.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/F4EB1DA6-E9AA-E811-8CF8-FA163ECCF441.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/A867DCA3-E9AA-E811-B8B3-FA163E1F69DA.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/DC0E29CC-E9AA-E811-AC0B-FA163E41F45F.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/8A5DAFA8-E9AA-E811-89C2-FA163EFF6119.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/B00828AF-E9AA-E811-8117-FA163E10FE53.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/28D7492E-E4AA-E811-9635-02163E013E90.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/FE2A5BAC-E9AA-E811-BC34-FA163E0639A2.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/7602E088-EDAA-E811-ABCA-FA163E749606.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/B49BB2AB-E9AA-E811-A0A0-FA163E3F57D2.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/14822C7E-EDAA-E811-8189-02163E01A095.root",
                                    "/store/data/Run2018D/SingleMuon/RAW/v1/000/321/909/00000/B05E9DA8-EDAA-E811-B422-02163E016528.root",
                            )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.p = cms.Path( 
                       (
                        process.rpcUnpackingModule 
                        + process.rpcTwinMuxRawToDigi 
                        + process.rpcCPPFRawToDigi 
                        + process.omtfStage2Digis
                        )
                       * process.rpcDigiMerger 
)

# Output
process.out = cms.OutputModule("PoolOutputModule"
                               , outputCommands = cms.untracked.vstring("drop *"
                                                                        , "keep *_*_*_testRPCDigiMerger")
                               # , fileName = cms.untracked.string(options.outputFile)
                               , fileName = cms.untracked.string("testRPCDigiMerger.root")
                               #, fileName = cms.untracked.string("testRPCDigiMerger.root")
                               , SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("p"))
)

process.e = cms.EndPath(process.out)
