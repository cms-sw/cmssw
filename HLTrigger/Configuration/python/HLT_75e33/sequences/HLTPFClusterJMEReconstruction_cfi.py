import FWCore.ParameterSet.Config as cms

from ..modules.hltAK4PFClusterJets_cfi import *
from ..modules.hltAK8PFClusterJets_cfi import *
from ..modules.hltPFClusterMET_cfi import *
from ..sequences.pfClusterRefsForJets_step_cfi import *

HLTPFClusterJMEReconstruction = cms.Sequence(
    pfClusterRefsForJets_step +
    hltAK4PFClusterJets +
    hltAK8PFClusterJets +
    hltPFClusterMET
)
# foo bar baz
# FBK0dGi6HoNnd
# VkYIOAtkAR8No
