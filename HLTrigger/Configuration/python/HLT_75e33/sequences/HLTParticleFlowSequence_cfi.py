import FWCore.ParameterSet.Config as cms

from ..sequences.caloTowersRecSequence_cfi import *
from ..sequences.iterTICLSequence_cfi import *
from ..sequences.particleFlowClusterSequence_cfi import *
from ..sequences.particleFlowRecoSequence_cfi import *
from ..sequences.particleFlowSuperClusteringSequence_cfi import *
from ..sequences.vertexRecoSequence_cfi import *

HLTParticleFlowSequence = cms.Sequence(particleFlowClusterSequence+iterTICLSequence+vertexRecoSequence+particleFlowSuperClusteringSequence+caloTowersRecSequence+particleFlowRecoSequence)
