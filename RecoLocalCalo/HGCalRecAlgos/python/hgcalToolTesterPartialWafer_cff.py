import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecAlgos.hgcalToolTesterPartialWaferEE_cfi import *

hgcalToolTesterPartialWaferHE = hgcalToolTesterPartialWaferEE.clone(
    nameSense  = "HGCalHESiliconSensitive",
    caloHitSource = "HGCHitsHEfront"
)
