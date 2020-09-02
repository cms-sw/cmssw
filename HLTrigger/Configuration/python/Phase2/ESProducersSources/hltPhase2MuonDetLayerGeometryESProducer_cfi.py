import FWCore.ParameterSet.Config as cms

from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import (
    MuonDetLayerGeometryESProducer as _MuonDetLayerGeometryESProducer,
)

hltPhase2MuonDetLayerGeometryESProducer = _MuonDetLayerGeometryESProducer.clone()
