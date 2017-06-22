import FWCore.ParameterSet.Config as cms

## configuration to build fast L1 ME0 trigger stubs
## pseudo pads are created from pseudo digis with 192 strips instead of 384
## the rechits are a necessary intermediate step before the pseudo pads are used
## as input to build pseudo stubs

from SimMuon.GEMDigitizer.muonME0PseudoReDigis_cfi import *
from RecoLocalMuon.GEMRecHit.me0RecHits_cfi import *
from RecoLocalMuon.GEMSegment.me0Segments_cfi import *

simMuonME0PseudoReDigisCoarse = simMuonME0PseudoReDigis.clone(
)
me0RecHitsCoarse = me0RecHits.clone(
    me0DigiLabel = cms.InputTag("simMuonME0PseudoReDigisCoarse")
)
nStrips = simMuonME0PseudoReDigisCoarse.numberOfStrips
me0TriggerPseudoDigis = me0Segments.clone(
    me0RecHitLabel = cms.InputTag("me0RecHitsCoarse")
)
me0TriggerPseudoDigis.algo_psets[1].algo_pset.maxPhiAdditional = cms.double(1.2*0.35/nStrips)
me0TriggerPseudoDigis.algo_psets[1].algo_pset.maxPhiSeeds = cms.double(1.2*0.35/nStrips)

me0TriggerPseudoDigiSequence = cms.Sequence(
    simMuonME0PseudoReDigisCoarse *
    me0RecHitsCoarse *
    me0TriggerPseudoDigis
)
