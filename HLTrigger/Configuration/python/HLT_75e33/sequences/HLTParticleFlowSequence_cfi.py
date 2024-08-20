import FWCore.ParameterSet.Config as cms

from ..sequences.HLTCaloTowersRecSequence_cfi import *
from ..sequences.HLTIterTICLSequence_cfi import *
from ..sequences.HLTParticleFlowClusterSequence_cfi import *
from ..sequences.HLTParticleFlowRecoSequence_cfi import *
from ..sequences.HLTParticleFlowSuperClusteringSequence_cfi import *
from ..sequences.HLTVertexRecoSequence_cfi import *

HLTParticleFlowSequence = cms.Sequence(HLTParticleFlowClusterSequence+HLTIterTICLSequence+HLTVertexRecoSequence+HLTParticleFlowSuperClusteringSequence+HLTCaloTowersRecSequence+HLTParticleFlowRecoSequence)
