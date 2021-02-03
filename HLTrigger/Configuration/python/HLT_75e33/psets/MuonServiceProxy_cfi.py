import FWCore.ParameterSet.Config as cms

MuonServiceProxy = cms.PSet(
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
            'StraightLinePropagator',
            'StraightLinePropagator'
        ),
        RPCLayers = cms.bool(True),
        UseMuonNavigation = cms.untracked.bool(True)
    )
)