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
        GEMLayers = cms.untracked.bool(False),
        ME0Layers = cms.bool(False),

        UseMuonNavigation = cms.untracked.bool(True)
    )
)

# run3_GEM
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(MuonServiceProxy,
    ServiceParameters = dict(GEMLayers = True)
)

# phase2_muon
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify(MuonServiceProxy,
    ServiceParameters = dict(ME0Layers = True)
)
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toModify(MuonServiceProxy,
    ServiceParameters = dict(ME0Layers = False)
)
