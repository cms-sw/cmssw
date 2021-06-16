import FWCore.ParameterSet.Config as cms

from RecoTracker.MkFit.mkFitHitConverterDefault_cfi import mkFitHitConverterDefault as _mkFitHitConverterDefault
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

mkFitHitConverter = _mkFitHitConverterDefault.clone(
    minGoodStripCharge = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
