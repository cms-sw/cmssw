import FWCore.ParameterSet.Config as cms

globalCosmicMuons = cms.EDProducer("GlobalCosmicMuonProducer",
    MuonTrackLoaderForCosmic,
    MuonServiceProxy,
    TrajectoryBuilderParameters = cms.PSet(
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        TkTrackCollectionLabel = cms.string('cosmictrackfinder'),
        Propagator = cms.string('SteppingHelixPropagatorAny'),
        SmootherParameters = cms.PSet(
            PropagatorAlong = cms.string('SteppingHelixPropagatorAlong'),
            PropagatorOpposite = cms.string('SteppingHelixPropagatorOpposite')
        )
    ),
    MuonCollectionLabel = cms.InputTag("cosmicMuons")
)


