import FWCore.ParameterSet.Config as cms
import EventFilter.RPCRawToDigi.RPCAMCRawToDigi_cfi as _mod

rpcCPPFRawToDigi = _mod.RPCAMCRawToDigi.clone(
    RPCAMCUnpacker = 'RPCCPPFUnpacker',
    RPCAMCUnpackerSettings = dict()
)
