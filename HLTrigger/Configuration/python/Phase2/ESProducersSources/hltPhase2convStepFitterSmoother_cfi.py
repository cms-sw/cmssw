import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.KFFittingSmoother_cfi import (
    KFFittingSmoother as _KFFittingSmoother,
)

hltPhase2convStepFitterSmoother = _KFFittingSmoother.clone(
    ComponentName="convStepFitterSmoother",
    EstimateCut=30,
    Fitter="RKFitter",
    MinNumberOfHits=3,
    Smoother="convStepRKSmoother",
)
