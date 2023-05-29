import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecAlgos.hgcalCheckToolDigiEE_cfi import *

hgcalCheckToolDigiHE = hgcalCheckToolDigiEE.clone(
    source = ('simHGCalUnsuppressedDigis', 'HEfront'),
    nameSense  = 'HGCalHESiliconSensitive',
)

hgcalCheckToolRecHitEE = hgcalCheckToolDigiEE.clone(
    source = ('HGCalRecHit', 'HGCEERecHits'),
    checkDigi = False,
)

hgcalCheckToolRecHitHE = hgcalCheckToolDigiEE.clone(
    source = ('HGCalRecHit',  'HGCHEFRecHits'),
    nameSense  = 'HGCalHESiliconSensitive',
    checkDigi = False,
)
