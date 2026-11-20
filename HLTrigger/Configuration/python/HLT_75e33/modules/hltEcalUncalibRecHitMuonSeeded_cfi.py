import FWCore.ParameterSet.Config as cms

from .hltEcalMultiFitUncalibRecHit_cfi import hltEcalMultiFitUncalibRecHit as _hltEcalMultiFitUncalibRecHit

hltEcalUncalibRecHitMuonSeeded = _hltEcalMultiFitUncalibRecHit.clone(
    EBdigiCollection = "hltEcalBarrelDigisMuonSeeded:ebDigis"
)
