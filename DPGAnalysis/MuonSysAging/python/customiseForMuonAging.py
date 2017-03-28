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

def appendAgingFileToGlobalTag(process):
    "Append aging DB file to GT from DPGAnalysis/MuonSysAging/test for testing purposes"

    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('MuonSystemAgingRcd'),
            tag = cms.string('MuonSystemAging_test'),
            connect = cms.string('sqlite_file://' 
                                 + os.environ['CMSSW_BASE'] 
                                 + '/src/DPGAnalysis/MuonSysAging/test/MuonSystemAging.db')
            )
        )

    return process

