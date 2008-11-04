import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonSegmentMatcher_cff import *

MuonTimingExtractorBlock = cms.PSet(
  timingParameters = cms.PSet(
    MuonSegmentMatcher,
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
            'PropagatorWithMaterial', 
            'PropagatorWithMaterialOpposite'),
        RPCLayers = cms.bool(True)
    ),
    DTsegments = cms.untracked.InputTag("dt4DSegments"),
    PruneCut = cms.double(0.3),
    HitsMin = cms.int32(3),
    debug = cms.bool(False),
  )
)


