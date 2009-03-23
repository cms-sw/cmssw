# AlCaReco for muon based alignment using beam-halo muons

import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlBeamHaloHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    eventSetupPathsKey = 'MuAlBeamHalo',
    throw = False
    )

ALCARECOMuAlBeamHalo = cms.EDFilter("AlignmentCSCBeamHaloSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("cosmicMuons"), # get cosmicMuons from global-run reconstruction
    minStations = cms.uint32(0), # no "energy cut" yet
    minHitsPerStation = cms.uint32(4)
)

seqALCARECOMuAlBeamHalo = cms.Sequence(ALCARECOMuAlBeamHaloHLT + ALCARECOMuAlBeamHalo)

