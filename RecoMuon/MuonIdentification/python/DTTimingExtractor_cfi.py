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
    DTsegments = cms.untracked.InputTag("dt4DSegments"),
    PruneCut = cms.double(1000.),
    HitsMin = cms.int32(3),
    UseSegmentT0 = cms.bool(False),
    RequireBothProjections = cms.bool(False),
    debug = cms.bool(False),
  )
)


