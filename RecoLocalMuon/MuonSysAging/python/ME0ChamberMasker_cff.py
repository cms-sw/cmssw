import FWCore.ParameterSet.Config as cms

from RecoLocalMuon.MuonSysAging.ME0ChamberMasker_cfi import ME0ChamberMasker
from SimMuon.GEMDigitizer.muonME0ReDigis_cfi import simMuonME0ReDigis

def appendME0ChamberMaskerAtReco(process):

    if hasattr(process,'RawToDigi') :

        print "[appendME0ChamberMasker] : Found RawToDigi, appending filter"

        process.simMuonME0Digis = ME0ChamberMasker.clone()
        process.simMuonME0ReDigis = simMuonME0ReDigis.clone()
        process.simMuonME0Digis.digiTag =  cms.InputTag("simMuonME0Digis", \
                                                        processName = cms.InputTag.skipCurrentProcess())

        process.filteredME0DigiSequence = cms.Sequence( process.simMuonME0Digis 
                                                        + process.simMuonME0ReDigis )

        process.RawToDigi += process.filteredME0DigiSequence

    return process


