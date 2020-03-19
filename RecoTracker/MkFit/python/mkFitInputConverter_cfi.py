import FWCore.ParameterSet.Config as cms

from RecoTracker.MkFit.mkFitInputConverterDefault_cfi import mkFitInputConverterDefault as _mkFitInputConverterDefault
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

mkFitInputConverter = _mkFitInputConverterDefault.clone(
    minGoodStripCharge = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutLoose')
    )
)
