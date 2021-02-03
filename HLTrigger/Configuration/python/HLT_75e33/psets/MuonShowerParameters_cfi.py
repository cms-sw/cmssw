import FWCore.ParameterSet.Config as cms

MuonShowerParameters = cms.PSet(
    MuonShowerInformationFillerParameters = cms.PSet(
        CSCRecSegmentLabel = cms.InputTag("csc2DRecHits"),
        CSCSegmentLabel = cms.InputTag("cscSegments"),
        DT4DRecSegmentLabel = cms.InputTag("dt4DSegments"),
        DTRecSegmentLabel = cms.InputTag("dt1DRecHits"),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
        ServiceParameters = cms.PSet(
            CSCLayers = cms.untracked.bool(True),
            GEMLayers = cms.untracked.bool(True),
            ME0Layers = cms.bool(True),
            Propagators = cms.untracked.vstring(
                'SteppingHelixPropagatorAny',
                'SteppingHelixPropagatorAlong',
                'SteppingHelixPropagatorOpposite',
                'SteppingHelixPropagatorL2Any',
                'SteppingHelixPropagatorL2Along',
                'SteppingHelixPropagatorL2Opposite',
                'SteppingHelixPropagatorAnyNoError',
                'SteppingHelixPropagatorAlongNoError',
                'SteppingHelixPropagatorOppositeNoError',
                'SteppingHelixPropagatorL2AnyNoError',
                'SteppingHelixPropagatorL2AlongNoError',
                'SteppingHelixPropagatorL2OppositeNoError',
                'PropagatorWithMaterial',
                'PropagatorWithMaterialOpposite',
                'SmartPropagator',
                'SmartPropagatorOpposite',
                'SmartPropagatorAnyOpposite',
                'SmartPropagatorAny',
                'SmartPropagatorRK',
                'SmartPropagatorAnyRK',
                'StraightLinePropagator'
            ),
            RPCLayers = cms.bool(True),
            UseMuonNavigation = cms.untracked.bool(True)
        ),
        TrackerRecHitBuilder = cms.string('WithTrackAngle')
    )
)