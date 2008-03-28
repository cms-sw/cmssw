import FWCore.ParameterSet.Config as cms

import copy
from ElectroWeakAnalysis.ZReco.zToEEOneTrack_cfi import *
#
# Reuse EWK modules, but reconfigure for higher mass cut
#
# This is the only real change
EgammaZOneTrack = copy.deepcopy(zToEEOneTrack)
import copy
from ElectroWeakAnalysis.ZReco.zToEEOneTrackGenParticlesMatch_cfi import *
# change the input tags to the changed module
EgammaZOneTrackGenParticlesMatch = copy.deepcopy(zToEEOneTrackGenParticlesMatch)
import copy
from ElectroWeakAnalysis.ZReco.zToEEOneTrackFilter_cfi import *
EgammaZOneTrackFilter = copy.deepcopy(zToEEOneTrackFilter)
EgammaZOneTrack.massMin = 60.
EgammaZOneTrackGenParticlesMatch.src = 'EgammaZOneTrack'
EgammaZOneTrackFilter.src = 'EgammaZOneTrack'

