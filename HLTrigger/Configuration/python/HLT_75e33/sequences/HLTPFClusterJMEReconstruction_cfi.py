import FWCore.ParameterSet.Config as cms

from ..modules.hltAK4PFClusterJets_cfi import *
from ..modules.hltAK8PFClusterJets_cfi import *
from ..modules.hltPFClusterMET_cfi import *
from ..sequences.pfClusterRefsForJetsSequence_cfi import *

HLTPFClusterJMEReconstruction = cms.Sequence(pfClusterRefsForJetsSequence+hltAK4PFClusterJets+hltAK8PFClusterJets+hltPFClusterMET)
