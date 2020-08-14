import FWCore.ParameterSet.Config as cms

from RecoTracker.MeasurementDet._MeasurementTrackerESProducer_default_cfi import (
    _MeasurementTrackerESProducer_default,
)

hltPhase2MeasurementTracker = _MeasurementTrackerESProducer_default.clone(
    Phase2StripCPE="Phase2StripCPE"
)
