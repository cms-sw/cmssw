import FWCore.ParameterSet.Config as cms

from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import (
    TrackerRecoGeometryESProducer as _TrackerRecoGeometryESProducer,
)

hltPhase2TrackerRecoGeometryESProducer = _TrackerRecoGeometryESProducer.clone(
    trackerGeometryLabel=cms.untracked.string("")
)
