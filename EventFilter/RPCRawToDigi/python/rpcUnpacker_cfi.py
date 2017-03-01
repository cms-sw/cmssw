import FWCore.ParameterSet.Config as cms
import EventFilter.RPCRawToDigi.rpcUnpackingModule_cfi

rpcunpacker =  EventFilter.RPCRawToDigi.rpcUnpackingModule_cfi.rpcUnpackingModule.clone()
rpcunpacker.InputLabel = cms.InputTag("rawDataCollector")
rpcunpacker.doSynchro = cms.bool(True)
