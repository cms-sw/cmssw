import FWCore.ParameterSet.Config as cms

from .hltEcalRecHit_cfi import hltEcalRecHit as _hltEcalRecHit

hltEcalRecHitMuonSeeded = _hltEcalRecHit.clone(
    EBuncalibRecHitCollection = "hltEcalUncalibRecHitMuonSeeded:EcalUncalibRecHitsEB",
    EEuncalibRecHitCollection = "hltEcalUncalibRecHitMuonSeeded:EcalUncalibRecHitsEE"
)
