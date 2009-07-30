import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonSegmentMatcher_cff import *

CSCTimingExtractorBlock = cms.PSet(
  CSCTimingParameters = cms.PSet(
    MuonSegmentMatcher,
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
            'PropagatorWithMaterial', 
            'PropagatorWithMaterialOpposite'),
        RPCLayers = cms.bool(True)
    ),
    CSCsegments = cms.InputTag("csc2DSegments"),
    PruneCut = cms.double(100.),
    CSCTimeOffset = cms.double(211.),
    debug = cms.bool(False),
  )
)


