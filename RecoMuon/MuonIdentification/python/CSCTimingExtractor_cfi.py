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
    PruneCut = cms.double(9.),
    CSCStripTimeOffset = cms.double(0.),
    CSCWireTimeOffset = cms.double(0.),
    CSCStripError = cms.double(7.0),
    CSCWireError = cms.double(8.6),
    # One of these next two lines must be true or no time is created
    UseStripTime = cms.bool(True),
    UseWireTime = cms.bool(True),
    debug = cms.bool(False)
  )
)


