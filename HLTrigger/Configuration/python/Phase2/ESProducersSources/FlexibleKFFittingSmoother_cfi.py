import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.FlexibleKFFittingSmoother_cfi import (
    FlexibleKFFittingSmoother as _FlexibleKFFittingSmoother,
)

hltPhase2FlexibleKFFittingSmoother = _FlexibleKFFittingSmoother.clone()
