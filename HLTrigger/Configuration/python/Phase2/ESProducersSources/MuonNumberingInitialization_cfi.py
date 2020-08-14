import FWCore.ParameterSet.Config as cms

from Geometry.MuonNumbering.muonNumberingInitialization_cfi import (
    MuonNumberingInitialization as _MuonNumberingInitialization,
)

hltPhase2MuonNumberingInitialization = _MuonNumberingInitialization.clone()
