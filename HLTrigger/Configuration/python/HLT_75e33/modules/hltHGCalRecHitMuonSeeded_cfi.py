import FWCore.ParameterSet.Config as cms

from .hltHGCalRecHit_cfi import hltHGCalRecHit as _hltHGCalRecHit

hltHGCalRecHitMuonSeeded = _hltHGCalRecHit.clone(
    HGCEEuncalibRecHitCollection = "hltHGCalUncalibRecHitMuonSeeded:HGCEEUncalibRecHits",
    HGCHEBuncalibRecHitCollection = "hltHGCalUncalibRecHitMuonSeeded:HGCHEBUncalibRecHits",
    HGCHEFuncalibRecHitCollection = "hltHGCalUncalibRecHitMuonSeeded:HGCHEFUncalibRecHits",
    HGCHFNoseuncalibRecHitCollection = "hltHGCalUncalibRecHitMuonSeeded:HGCHFNoseUncalibRecHits"
)
