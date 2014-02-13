import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterECALNew import *

particleFlowClusterECALNewSequence = cms.Sequence(
    particleFlowClusterECALBarrel +
    particleFlowClusterECALEndcap +
    particleFlowClusterECALNew
    )
                                                   
