import FWCore.ParameterSet.Config as cms

from DPGAnalysis.MuonSysAging.RPCChamberMasker_cfi import RPCChamberMasker

def appendRPCChamberMaskerAtUnpacking2(process):
    print "[appendChamberMasker] : Found muonRPCDigis, applying filter"
    process.simMuonRPCDigis = RPCChamberMasker.clone()
    process.simMuonRPCDigis.digiTag =  cms.InputTag("simMuonRPCDigis", processName = cms.InputTag.skipCurrentProcess())
    process.filteredRPCDigiSequence = cms.Sequence( process.simMuonRPCDigis)
    process.RawToDigi += process.filteredRPCDigiSequence

    return process

