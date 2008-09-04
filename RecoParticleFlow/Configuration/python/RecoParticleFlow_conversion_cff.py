import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.towerMakerPF_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *

from RecoParticleFlow.PFTracking.particleFlowTrackWithConversion_cff import *

from RecoParticleFlow.PFBlockProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowBlock_cff import *

from RecoParticleFlow.PFProducer.particleFlow_cff import *

particleFlowRecoConversion = cms.Sequence( caloTowersPFRec*
                                           particleFlowCluster*
                                           particleFlowTrackWithConversion*
                                           particleFlowBlock*
                                           particleFlow )

