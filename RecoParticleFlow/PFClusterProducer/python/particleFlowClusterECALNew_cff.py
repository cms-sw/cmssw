import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterECALNew_cfi import *

particleFlowClusterECALNewSequence = cms.Sequence(
    particleFlowClusterECALBarrel +
    particleFlowClusterECALEndcap +
    particleFlowClusterECALNew
    )
                                                   
