import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for muon based alignment using beam-halo muons in the CSC overlap regions
ALCARECOMuAlOverlapsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlOverlapsMuonSelector = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone()
ALCARECOMuAlOverlaps = cms.EDFilter("AlignmentCSCOverlapSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("ALCARECOMuAlOverlapsMuonSelector","StandAlone"),
    minHitsPerChamber = cms.uint32(4),
    station = cms.int32(0) ## all stations: I'll need to split it by station (8 subsamples) offline

)

seqALCARECOMuAlOverlaps = cms.Sequence(ALCARECOMuAlOverlapsHLT+ALCARECOMuAlOverlapsMuonSelector*ALCARECOMuAlOverlaps)
ALCARECOMuAlOverlapsHLT.HLTPaths = ['HLT1MuonPrescalePt3', 'HLT1MuonPrescalePt5', 'HLT1MuonIso', 'HLT1MuonNonIso']
ALCARECOMuAlOverlapsMuonSelector.ptMin = 3.

