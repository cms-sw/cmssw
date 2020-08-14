import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits._generic_default_cfi import _generic_default

hltPhase2PixelCPEGenericESProducer = _generic_default.clone(
    MagneticFieldRecord=cms.ESInputTag("", ""),
    NoTemplateErrorsWhenNoTrkAngles=True,
    TruncatePixelCharge=False,
    Upgrade=True,
)
