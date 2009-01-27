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
        TkTrackCollectionLabel = cms.string('generalTracks'),
        Propagator = cms.string('SteppingHelixPropagatorAny'),
        SmootherParameters = cms.PSet(
            PropagatorAlong = cms.string('SmartPropagatorAny'),
            PropagatorOpposite = cms.string('SmartPropagatorAnyOpposite')
        )
    ),
    MuonCollectionLabel = cms.InputTag("cosmicMuons")
)


