# AlCaReco for muon based alignment using beam-halo muons in the CSC overlap regions
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlOverlapsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlOverlaps',
    throw = False # tolerate triggers not available
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

seqALCARECOMuAlOverlaps = cms.Sequence(ALCARECOMuAlOverlapsHLT+ALCARECOMuAlOverlapsMuonSelector*ALCARECOMuAlOverlaps)
