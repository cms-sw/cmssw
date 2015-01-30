import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonSegmentMatcher_cff import *

DTTimingExtractorBlock = cms.PSet(
  DTTimingParameters = cms.PSet(
    MuonSegmentMatcher,
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
            'PropagatorWithMaterial', 
            'PropagatorWithMaterialOpposite'),
        RPCLayers = cms.bool(True)
    ),
    PruneCut = cms.double(10000.),
    DTTimeOffset = cms.double(0.),
    HitError  = cms.double(6.0),
    HitsMin = cms.int32(3),
    UseSegmentT0 = cms.bool(False),
    DoWireCorr = cms.bool(True),
    DropTheta = cms.bool(True),
    RequireBothProjections = cms.bool(False),
    debug = cms.bool(False),
  )
)


