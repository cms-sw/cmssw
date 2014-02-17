import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cfi import *

particleFlowClusterECALSequence = cms.Sequence(
    particleFlowClusterECALUncorrected +
    particleFlowClusterECAL
    )
                                                   
