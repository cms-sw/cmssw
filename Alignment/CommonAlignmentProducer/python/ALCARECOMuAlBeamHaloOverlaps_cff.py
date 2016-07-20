# AlCaReco for muon based alignment using beam-halo muons in the CSC overlap regions

import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlBeamHaloOverlapsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    eventSetupPathsKey = 'MuAlBeamHaloOverlaps',
    throw = False # tolerate triggers not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOMuAlBeamHaloOverlapsDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(False),
    DebugOn      = cms.untracked.bool(False)
)

ALCARECOMuAlBeamHaloOverlapsEnergyCut = cms.EDFilter("AlignmentCSCBeamHaloSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("cosmicMuons"), # get cosmicMuons from global-run reconstruction
    minStations = cms.uint32(0), # no "energy cut" yet
    minHitsPerStation = cms.uint32(1)
)

ALCARECOMuAlBeamHaloOverlaps = cms.EDFilter("AlignmentCSCOverlapSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("ALCARECOMuAlBeamHaloOverlapsEnergyCut"),
    minHitsPerChamber = cms.uint32(1),
    station = cms.int32(0) # all stations: I'll need to split it by station (8 subsamples) offline
)

seqALCARECOMuAlBeamHaloOverlaps = cms.Sequence(ALCARECOMuAlBeamHaloOverlapsHLT + ALCARECOMuAlBeamHaloOverlapsDCSFilter + ALCARECOMuAlBeamHaloOverlapsEnergyCut * ALCARECOMuAlBeamHaloOverlaps)
