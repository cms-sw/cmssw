import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecAlgos.hgcalCheckToolDigiEE_cfi import *

hgcalCheckToolDigiHE = hgcalCheckToolDigiEE.clone(
    source = cms.InputTag('simHGCalUnsuppressedDigis', 'HEfront'),
    nameSense  = "HGCalHESiliconSensitive",
)

hgcalCheckToolRecHitEE = hgcalCheckToolDigiEE.clone(
    source = cms.InputTag('HGCalRecHit', 'HGCEERecHits'),
    checkDigi = False,
)

hgcalCheckToolRecHitHE = hgcalCheckToolDigiEE.clone(
    source = cms.InputTag('HGCalRecHit', 'HGCHEFRecHits'),
    nameSense  = "HGCalHESiliconSensitive",
    checkDigi = False,
)
