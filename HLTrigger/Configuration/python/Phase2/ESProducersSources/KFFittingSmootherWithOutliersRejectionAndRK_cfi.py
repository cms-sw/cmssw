import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.KFFittingSmoother_cfi import (
    KFFittingSmoother as _KFFittingSmoother,
)

hltPhase2KFFittingSmootherWithOutliersRejectionAndRK = _KFFittingSmoother.clone(
    BreakTrajWith2ConsecutiveMissing=True,
    ComponentName="KFFittingSmootherWithOutliersRejectionAndRK",
    EstimateCut=20.0,
    Fitter="RKFitter",
    MinNumberOfHits=3,
    Smoother="RKSmoother",
)
