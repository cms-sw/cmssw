import FWCore.ParameterSet.Config as cms

from ..modules.particleFlowBadHcalPseudoCluster_cfi import *
from ..sequences.pfClusteringECALSequence_cfi import *
from ..sequences.pfClusteringHBHEHFSequence_cfi import *
from ..sequences.pfClusteringHOSequence_cfi import *

particleFlowClusterSequence = cms.Sequence(particleFlowBadHcalPseudoCluster+pfClusteringECALSequence+pfClusteringHBHEHFSequence+pfClusteringHOSequence)
