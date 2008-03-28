import FWCore.ParameterSet.Config as cms

import copy
from ElectroWeakAnalysis.ZReco.zToEEOneSuperCluster_cfi import *
#
# Reuse EWK modules, but reconfigure for higher mass cut
#
# This is the only real change
EgammaZOneSuperCluster = copy.deepcopy(zToEEOneSuperCluster)
import copy
from ElectroWeakAnalysis.ZReco.zToEEOneSuperClusterGenParticlesMatch_cfi import *
# change the input tags to the changed module
EgammaZOneSuperClusterGenParticlesMatch = copy.deepcopy(zToEEOneSuperClusterGenParticlesMatch)
import copy
from ElectroWeakAnalysis.ZReco.zToEEOneSuperClusterFilter_cfi import *
EgammaZOneSuperClusterFilter = copy.deepcopy(zToEEOneSuperClusterFilter)
EgammaZOneSuperCluster.massMin = 60.
EgammaZOneSuperClusterGenParticlesMatch.src = 'EgammaZOneSuperCluster'
EgammaZOneSuperClusterFilter.src = 'EgammaZOneSuperCluster'

