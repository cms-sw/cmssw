import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
#PropagatorWithMaterialESProducer 
oppositeToMomElePropagator = copy.deepcopy(OppositeMaterialPropagator)
oppositeToMomElePropagator.Mass = 0.000511
oppositeToMomElePropagator.ComponentName = 'oppositeToMomElePropagator'

