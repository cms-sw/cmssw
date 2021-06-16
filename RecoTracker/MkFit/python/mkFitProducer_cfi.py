import FWCore.ParameterSet.Config as cms

from RecoTracker.MkFit.mkFitProducerDefault_cfi import mkFitProducerDefault as _mkFitProducerDefault
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

mkFitProducer = _mkFitProducerDefault.clone(
    minGoodStripCharge = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
