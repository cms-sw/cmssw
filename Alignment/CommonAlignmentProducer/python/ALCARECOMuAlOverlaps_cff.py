# AlCaReco for muon based alignment using beam-halo muons in the CSC overlap regions
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlOverlapsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlOverlaps',
    throw = False # tolerate triggers not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOMuAlOverlapsDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(False),
    DebugOn      = cms.untracked.bool(False)
)

ALCARECOMuAlOverlaps = cms.EDFilter("AlignmentCSCOverlapSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("ALCARECOMuAlOverlapsMuonSelector","StandAlone"),
    minHitsPerChamber = cms.uint32(4),
    station = cms.int32(0) ## all stations: the algorithm can handle multiple stations now
)

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlOverlapsMuonSelector = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    ptMin = 3.
    )

seqALCARECOMuAlOverlaps = cms.Sequence(ALCARECOMuAlOverlapsHLT+ALCARECOMuAlOverlapsDCSFilter+ALCARECOMuAlOverlapsMuonSelector*ALCARECOMuAlOverlaps)
