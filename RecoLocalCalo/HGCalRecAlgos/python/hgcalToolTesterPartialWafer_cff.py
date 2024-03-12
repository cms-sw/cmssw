import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecAlgos.hgcalToolTesterPartialWaferEE_cfi import *

hgcalToolTesterPartialWaferHE = hgcalToolTesterPartialWaferEE.clone(
    nameSense  = "HGCalHESiliconSensitive",
    caloHitSource = "HGCHitsHEfront"
)
# foo bar baz
# 3eYn7FhnLwyaU
# MCfNykJc2Fw7d
