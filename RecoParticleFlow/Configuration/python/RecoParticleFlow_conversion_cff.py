import FWCore.ParameterSet.Config as cms


from RecoParticleFlow.PFTracking.particleFlowTrackWithConversion_cff import *

from RecoParticleFlow.PFProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFProducer.particleFlowBlock_cff import *
particleFlowBlock.useConversions =True

from RecoParticleFlow.PFProducer.particleFlow_cff import *

particleFlowRecoConversionTask = cms.Task( particleFlowTrackWithConversionTask,
                                           particleFlowBlock,
                                           particleFlowTmp )
particleFlowRecoConversion = cms.Sequence(particleFlowRecoConversionTask)
