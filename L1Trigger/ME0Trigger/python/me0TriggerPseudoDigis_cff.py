import FWCore.ParameterSet.Config as cms

## configuration to build fast L1 ME0 trigger stubs
## pseudo pads are created from pseudo digis with 192 strips instead of 384
## the rechits are a necessary intermediate step before the pseudo pads are used
## as input to build pseudo stubs

from SimMuon.GEMDigitizer.muonME0PseudoReDigis_cfi import *
from RecoLocalMuon.GEMRecHit.me0RecHits_cfi import *
from RecoLocalMuon.GEMSegment.me0Segments_cfi import *

simMuonME0PseudoReDigis192 = simMuonME0PseudoReDigis.clone(
    numberOfStrips = cms.uint32(192)
)
me0RecHits192 = me0RecHits.clone(
    me0DigiLabel = cms.InputTag("simMuonME0PseudoReDigis192")
)
me0TriggerPseudoDigis = me0Segments.clone(
    me0RecHitLabel = cms.InputTag("me0RecHits192")
)
me0TriggerPseudoDigiSequence = cms.Sequence(
    simMuonME0PseudoReDigis192 *
    me0RecHits192 *
    me0TriggerPseudoDigis
)
