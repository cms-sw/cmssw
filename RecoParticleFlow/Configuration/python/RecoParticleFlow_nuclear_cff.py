import FWCore.ParameterSet.Config as cms


from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *

from RecoParticleFlow.PFBlockProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowBlock_cff import *

from RecoParticleFlow.PFProducer.particleFlow_cff import *
from RecoParticleFlow.PFTracking.nuclearRemainingHits_cff import *

particleFlowRecoNuclear = cms.Sequence(
    nuclearRemainingHits*
    particleFlowTrackWithNuclear*
    particleFlowBlock*particleFlow
    )
particleFlowBlock.useNuclear = True

 
