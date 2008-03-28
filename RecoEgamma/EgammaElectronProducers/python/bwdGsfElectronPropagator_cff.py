import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
# "backward" propagator for electrons
bwdGsfElectronPropagator = copy.deepcopy(OppositeMaterialPropagator)
bwdGsfElectronPropagator.Mass = 0.000511
bwdGsfElectronPropagator.ComponentName = 'bwdGsfElectronPropagator'

