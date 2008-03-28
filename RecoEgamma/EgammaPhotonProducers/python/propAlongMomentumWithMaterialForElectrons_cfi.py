import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
#PropagatorWithMaterialESProducer 
alongMomElePropagator = copy.deepcopy(MaterialPropagator)
alongMomElePropagator.Mass = 0.000511
alongMomElePropagator.ComponentName = 'alongMomElePropagator'

