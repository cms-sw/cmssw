import FWCore.ParameterSet.Config as cms

from RecoTracker.MkFit.mkFitSiStripHitConverterDefault_cfi import mkFitSiStripHitConverterDefault as _mkFitSiStripHitConverterDefault
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

mkFitSiStripHitConverter = _mkFitSiStripHitConverterDefault.clone(
    minGoodStripCharge = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
