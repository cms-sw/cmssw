import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# "forward" propagator for electrons
fwdGsfElectronPropagator = copy.deepcopy(MaterialPropagator)
fwdGsfElectronPropagator.Mass = 0.000511
fwdGsfElectronPropagator.ComponentName = 'fwdGsfElectronPropagator'

