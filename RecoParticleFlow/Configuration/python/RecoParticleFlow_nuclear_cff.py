import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *

from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *

from RecoParticleFlow.PFBlockProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowBlock_cff import *

from RecoParticleFlow.PFProducer.particleFlow_cff import *
from RecoParticleFlow.PFTracking.nuclearRemainingHits_cff import *

particleFlowRecoNuclear = cms.Sequence(
    nuclearRemainingHits*
    particleFlowCluster*
    particleFlowTrackWithNuclear*
    particleFlowBlock*particleFlow
    )
particleFlowBlock.useNuclear = True

 
