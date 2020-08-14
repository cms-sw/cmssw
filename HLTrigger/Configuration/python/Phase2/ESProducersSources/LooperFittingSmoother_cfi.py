import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.KFFittingSmoother_cfi import (
    KFFittingSmoother as _KFFittingSmoother,
)

hltPhase2LooperFittingSmoother = _KFFittingSmoother.clone(
    ComponentName="LooperFittingSmoother",
    EstimateCut=20.0,
    Fitter="LooperFitter",
    LogPixelProbabilityCut=-14.0,
    MinNumberOfHits=3,
    Smoother="LooperSmoother",
)
