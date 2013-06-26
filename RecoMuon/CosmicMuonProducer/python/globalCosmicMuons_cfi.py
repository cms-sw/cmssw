import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
from RecoMuon.GlobalTrackingTools.GlobalMuonTrackMatcher_cff import *

globalCosmicMuons = cms.EDProducer("GlobalCosmicMuonProducer",
    MuonTrackLoaderForCosmic,
    MuonServiceProxy,
    TrajectoryBuilderParameters = cms.PSet(
        GlobalMuonTrackMatcher,
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        TkTrackCollectionLabel = cms.InputTag("generalTracks"),
        Propagator = cms.string('SteppingHelixPropagatorAny'),
        SmootherParameters = cms.PSet(
            PropagatorAlong = cms.string('SteppingHelixPropagatorAny'),
            PropagatorOpposite = cms.string('SteppingHelixPropagatorAny'),
            RescalingFactor = cms.double(5.0)
        )
    ),
    MuonCollectionLabel = cms.InputTag("cosmicMuons")
)


