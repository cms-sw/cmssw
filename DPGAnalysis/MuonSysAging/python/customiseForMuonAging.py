import FWCore.ParameterSet.Config as cms

import os

from DPGAnalysis.MuonSysAging.DTChamberMasker_cff  import *
from DPGAnalysis.MuonSysAging.RPCChamberMasker_cff import *
from DPGAnalysis.MuonSysAging.CSCChamberMasker_cff import *
from DPGAnalysis.MuonSysAging.GEMChamberMasker_cff import *
from DPGAnalysis.MuonSysAging.ME0ChamberMasker_cff import *

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon

def enableAgingAtReco(process):
    "Enable muon masking/aging for digis entering muon reconstruction"    
  

    appendCSCChamberMaskerAtUnpacking(process)
    appendDTChamberMaskerAtUnpacking(process) 
    if phase2_muon.isChosen():
        appendRPCChamberMaskerBeforeRecHits(process)
        appendGEMChamberMaskerAtReco(process)    
        appendME0ChamberMaskerAtReco(process)    
    else :
        appendRPCChamberMaskerAtUnpacking(process)

    return process
