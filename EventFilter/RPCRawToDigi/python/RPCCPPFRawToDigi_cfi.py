import FWCore.ParameterSet.Config as cms
import EventFilter.RPCRawToDigi.RPCAMCRawToDigi_cfi as _mod

rpcCPPFRawToDigi = _mod.RPCAMCRawToDigi.clone(
    RPCAMCUnpacker = 'RPCCPPFUnpacker',
    RPCAMCUnpackerSettings = dict()
)

from Configuration.Eras.Modifier_run3_RPC_cff import run3_RPC
run3_RPC.toModify(rpcCPPFRawToDigi, 
        RPCAMCUnpackerSettings = dict(cppfDaqDelay = 2)
)
