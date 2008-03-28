import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for muon based alignment using beam-halo muons in the CSC overlap regions
ALCARECOMuAlOverlapsHLT = copy.deepcopy(hltHighLevel)
import copy
from Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi import *
ALCARECOMuAlOverlapsMuonSelector = copy.deepcopy(AlignmentMuonSelector)
ALCARECOMuAlOverlaps = cms.EDFilter("AlignmentCSCOverlapSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("ALCARECOMuAlOverlapsMuonSelector","GlobalMuon"),
    minHitsPerChamber = cms.uint32(4),
    station = cms.int32(0) ## all stations: I'll need to split it by station (8 subsamples) offline

)

seqALCARECOMuAlOverlaps = cms.Sequence(ALCARECOMuAlOverlapsHLT+ALCARECOMuAlOverlapsMuonSelector*ALCARECOMuAlOverlaps)
ALCARECOMuAlOverlapsHLT.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso']
ALCARECOMuAlOverlapsMuonSelector.ptMin = 3.

