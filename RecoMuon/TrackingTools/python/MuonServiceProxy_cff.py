import FWCore.ParameterSet.Config as cms


from TrackingTools.GeomPropagators.StraightLinePropagator_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorHLT_cff import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorsNoErrorPropagation_cff import *
from TrackingTools.GeomPropagators.SmartPropagator_cff import *
from TrackingTools.GeomPropagators.SmartPropagatorAnyOpposite_cfi import *
from TrackingTools.GeomPropagators.SmartPropagatorAny_cfi import *
MuonServiceProxy = cms.PSet(
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
                                            'SteppingHelixPropagatorAlong', 
                                            'SteppingHelixPropagatorOpposite', 
                                            'SteppingHelixPropagatorL2Any', 
                                            'SteppingHelixPropagatorL2Along', 
                                            'SteppingHelixPropagatorL2Opposite', 
                                            #A bunch with error propagation turned off: can be about 2 times faster 
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
                                            'StraightLinePropagator'),
        RPCLayers = cms.bool(True),
        CSCLayers = cms.untracked.bool(True),
        GEMLayers = cms.untracked.bool(True),
        UseMuonNavigation = cms.untracked.bool(True)
    )
)


