import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *

globalCosmicMuons = cms.EDProducer("GlobalCosmicMuonProducer",
    MuonTrackLoaderForCosmic,
    MuonServiceProxy,
    TrajectoryBuilderParameters = cms.PSet(
        GlobalMuonTrackMatcher = cms.PSet(
          MinP = cms.double(2.5),
          MinPt = cms.double(1.0),
          Pt_threshold= cms.double(35.0),
          Eta_threshold= cms.double(1.0),
          Chi2Cut= cms.double(50.0),
          LocChi2Cut= cms.double(.008),
          DeltaDCut= cms.double(10.0),
          DeltaRCut= cms.double(.2),
          Propagator = cms.string('SteppingHelixPropagatorAny')
        ),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        TkTrackCollectionLabel = cms.InputTag("generalTracks"),
        Propagator = cms.string('SteppingHelixPropagatorAny'),
        SmootherParameters = cms.PSet(
            PropagatorAlong = cms.string('SmartPropagatorAny'),
            PropagatorOpposite = cms.string('SmartPropagatorAnyOpposite'),
            RescalingFactor = cms.double(5.0)
        )
    ),
    MuonCollectionLabel = cms.InputTag("cosmicMuons")
)


