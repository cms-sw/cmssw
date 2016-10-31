# AlCaReco for muon based alignment using beam-halo muons

import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlBeamHaloHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    eventSetupPathsKey = 'MuAlBeamHalo',
    throw = False
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOMuAlBeamHaloDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(False),
    DebugOn      = cms.untracked.bool(False)
)

ALCARECOMuAlBeamHalo = cms.EDFilter("AlignmentCSCBeamHaloSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("cosmicMuons"), # get cosmicMuons from global-run reconstruction
    minStations = cms.uint32(0), # no "energy cut" yet
    minHitsPerStation = cms.uint32(1)
)

seqALCARECOMuAlBeamHalo = cms.Sequence(ALCARECOMuAlBeamHaloHLT + ALCARECOMuAlBeamHaloDCSFilter + ALCARECOMuAlBeamHalo)

