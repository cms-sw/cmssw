import FWCore.ParameterSet.Config as cms

from .hltHGCalUncalibRecHit_cfi import hltHGCalUncalibRecHit as _hltHGCalUncalibRecHit

hltHGCalUncalibRecHitMuonSeeded = _hltHGCalUncalibRecHit.clone(
    HGCEEdigiCollection = "hltHgcalDigisMuonSeeded:EE",
    HGCHEBdigiCollection = "hltHgcalDigisMuonSeeded:HEback",
    HGCHEFdigiCollection = "hltHgcalDigisMuonSeeded:HEfront"
)
