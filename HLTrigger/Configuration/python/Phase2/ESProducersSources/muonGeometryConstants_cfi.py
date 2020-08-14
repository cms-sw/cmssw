import FWCore.ParameterSet.Config as cms

from Geometry.MuonNumbering.muonGeometryConstants_cfi import (
    muonGeometryConstants as _muonGeometryConstants,
)

hltPhase2muonGeometryConstants = _muonGeometryConstants.clone()
