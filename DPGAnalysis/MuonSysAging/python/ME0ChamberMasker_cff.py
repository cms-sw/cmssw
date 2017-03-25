import FWCore.ParameterSet.Config as cms

from DPGAnalysis.MuonSysAging.ME0ChamberMasker_cfi import ME0ChamberMasker


def appendME0ChamberMaskerAtUnpacking2(process):
    print "[appendChamberMasker] : Found muonME0Digis, applying filter"
    process.simMuonME0Digis = ME0ChamberMasker.clone()
    process.simMuonME0Digis.digiTag =  cms.InputTag("simMuonME0Digis", processName = cms.InputTag.skipCurrentProcess())
    process.filteredME0DigiSequence = cms.Sequence( process.simMuonME0Digis)
    process.RawToDigi += process.filteredME0DigiSequence
    return process


