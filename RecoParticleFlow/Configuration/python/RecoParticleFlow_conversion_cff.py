import FWCore.ParameterSet.Config as cms


from RecoParticleFlow.PFTracking.particleFlowTrackWithConversion_cff import *

from RecoParticleFlow.PFBlockProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowBlock_cff import *
particleFlowBlock.useConversions =True

from RecoParticleFlow.PFProducer.particleFlow_cff import *

particleFlowRecoConversion = cms.Sequence( particleFlowTrackWithConversion*
                                           particleFlowBlock*
                                           particleFlow )

