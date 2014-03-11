import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECALWithTime_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECALWithTime_cfi import *

particleFlowClusterECALSequence = cms.Sequence(
    particleFlowRecHitECALWithTime+
    particleFlowClusterECALWithTimeUncorrected +
    particleFlowClusterECALWithTimeSelected +
    particleFlowClusterECAL
    )
                                                   
