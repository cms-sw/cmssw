import FWCore.ParameterSet.Config as cms

from ..modules.hltParticleFlowBadHcalPseudoCluster_cfi import *
from ..sequences.HLTPfClusteringECALSequence_cfi import *
from ..sequences.HLTPfClusteringHBHEHFSequence_cfi import *
from ..sequences.HLTPfClusteringHOSequence_cfi import *

HLTParticleFlowClusterSequence = cms.Sequence(hltParticleFlowBadHcalPseudoCluster+HLTPfClusteringECALSequence+HLTPfClusteringHBHEHFSequence+HLTPfClusteringHOSequence)
