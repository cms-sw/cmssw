import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import (
    KFUpdatorESProducer as _KFUpdatorESProducer,
)

hltPhase2KFUpdatorESProducer = _KFUpdatorESProducer.clone()
