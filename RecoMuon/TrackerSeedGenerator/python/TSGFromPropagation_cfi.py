import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *

SeedGeneratorParameters = cms.PSet(
    MuonErrorMatrixValues,
    ComponentName = cms.string( "TSGFromPropagation" ),
    Propagator = cms.string( "SmartPropagatorAnyOpposite" ),
    MaxChi2 = cms.double( 40.0 ),
    ResetMethod = cms.string("matrix"),
    ErrorRescaling = cms.double(3.0),
    SigmaZ = cms.double(25.0),
    UseVertexState = cms.bool( True ),
    UpdateState = cms.bool( True ),
    SelectState = cms.bool( False ),
    beamSpot = cms.InputTag("offlineBeamSpot")
    )
