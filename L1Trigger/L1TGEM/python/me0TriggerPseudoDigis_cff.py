import FWCore.ParameterSet.Config as cms

## configuration to build fast L1 ME0 trigger stubs
## pseudo pads are created from pseudo digis with 192 strips instead of 384
## the rechits are a necessary intermediate step before the pseudo pads are used
## as input to build pseudo stubs

from SimMuon.GEMDigitizer.muonME0PseudoReDigis_cfi import *
from RecoLocalMuon.GEMRecHit.me0RecHits_cfi import *
from RecoLocalMuon.GEMSegment.me0Segments_cfi import *
from L1Trigger.L1TGEM.me0TriggerConvertedPseudoDigis_cfi import *

simMuonME0PseudoReDigisCoarse = simMuonME0PseudoReDigis.clone(
    usePads = cms.bool(True)
)
me0RecHitsCoarse = me0RecHits.clone(
    me0DigiLabel = cms.InputTag("simMuonME0PseudoReDigisCoarse")
)

me0TriggerPseudoDigis = me0Segments.clone(
    me0RecHitLabel = cms.InputTag("me0RecHitsCoarse")
)
## 1.2 is to make the matching window safely the two nearest strips
## 0.35 is the size of an ME0 chamber in radians
## nStrips is divided by 2 since we use 2-strip trigger pads
nStrips = simMuonME0PseudoReDigisCoarse.numberOfStrips.value()//2
maxPhi = 1.2*0.35/nStrips
me0TriggerPseudoDigis.algo_psets[1].algo_pset.maxPhiAdditional = cms.double(maxPhi)
me0TriggerPseudoDigis.algo_psets[1].algo_pset.maxPhiSeeds = cms.double(maxPhi)

me0TriggerPseudoDigiTask = cms.Task(
    simMuonME0PseudoReDigisCoarse,
    me0RecHitsCoarse,
    me0TriggerPseudoDigis,
    ## need to run the standard ME0 RECO sequence for converted triggers
    me0RecHits,
    me0Segments,
    me0TriggerConvertedPseudoDigis
)

from RecoLocalMuon.GEMRecHit.gemRecHits_cfi import *
from RecoLocalMuon.GEMSegment.gemSegments_cfi import *

ge0TriggerPseudoDigiTask = cms.Task(
    simMuonME0PseudoReDigisCoarse,
    me0RecHitsCoarse,
    me0TriggerPseudoDigis,
    ## need to run the standard ME0 RECO sequence for converted triggers
    gemRecHits,
    gemSegments,
    ge0TriggerConvertedPseudoDigis
)
