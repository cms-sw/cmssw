import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
globalCosmicMuons = cms.EDProducer("GlobalCosmicMuonProducer",
    MuonTrackLoaderForCosmic,
    MuonServiceProxy,
    TrajectoryBuilderParameters = cms.PSet(
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        TkTrackCollectionLabel = cms.string('generalTracks'),
        Propagator = cms.string('SteppingHelixPropagatorAny'),
        SmootherParameters = cms.PSet(
            PropagatorAlong = cms.string('SmartPropagatorAny'),
            PropagatorOpposite = cms.string('SmartPropagatorAnyOpposite')
        ),
        GlobalMuonTrackMatcher = cms.PSet(
            MinP = cms.double(2.5),
            Chi2Cut = cms.double(50.0),
            MinPt = cms.double(1.0),
            DeltaDCut = cms.double(10.0),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            DeltaRCut = cms.double(0.2)
        ) 
    ),
    MuonCollectionLabel = cms.InputTag("cosmicMuons")
)


