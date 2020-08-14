import FWCore.ParameterSet.Config as cms

from TrackingTools.RecoGeometry.GlobalDetLayerGeometryESProducer_cfi import (
    GlobalDetLayerGeometry as _GlobalDetLayerGeometry,
)

hltPhase2GlobalDetLayerGeometry = _GlobalDetLayerGeometry.clone()
