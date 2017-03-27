import FWCore.ParameterSet.Config as cms

from DPGAnalysis.MuonSysAging.GEMChamberMasker_cfi import GEMChamberMasker
from SimMuon.GEMDigitizer.muonGEMPadDigis_cfi import simMuonGEMPadDigis
from SimMuon.GEMDigitizer.muonGEMPadDigiClusters_cfi import simMuonGEMPadDigiClusters


def appendGEMChamberMaskerAtUnpacking2(process):
    print "[appendChamberMasker] : Found muonGEMDigis, applying filter"
    process.simMuonGEMPadDigis = simMuonGEMPadDigis.clone()
    process.simMuonGEMPadDigiClusters = simMuonGEMPadDigiClusters.clone()
    process.simMuonGEMDigis = GEMChamberMasker.clone()
    process.simMuonGEMDigis.digiTag =  cms.InputTag("simMuonGEMDigis", processName = cms.InputTag.skipCurrentProcess())
    process.filteredGEMDigiSequence = cms.Sequence( process.simMuonGEMDigis*process.simMuonGEMPadDigis*process.simMuonGEMPadDigiClusters)
    process.RawToDigi += process.filteredGEMDigiSequence
    return process


