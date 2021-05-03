import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowClusterHBHEForEgamma_cfi import *
from ..modules.hltParticleFlowClusterHCALForEgamma_cfi import *
from ..modules.hltParticleFlowRecHitHBHEForEgamma_cfi import *
from ..modules.hltRegionalTowerForEgamma_cfi import *

HLTPFHcalClusteringForEgammaTask = cms.Task(hltParticleFlowClusterHBHEForEgamma, hltParticleFlowClusterHCALForEgamma, hltParticleFlowRecHitHBHEForEgamma, hltRegionalTowerForEgamma)
