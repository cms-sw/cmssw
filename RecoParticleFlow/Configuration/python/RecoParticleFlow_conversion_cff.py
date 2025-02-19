import FWCore.ParameterSet.Config as cms


from RecoParticleFlow.PFTracking.particleFlowTrackWithConversion_cff import *

from RecoParticleFlow.PFProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFProducer.particleFlowBlock_cff import *
particleFlowBlock.useConversions =True

from RecoParticleFlow.PFProducer.particleFlow_cff import *

particleFlowRecoConversion = cms.Sequence( particleFlowTrackWithConversion*
                                           particleFlowBlock*
                                           particleFlowTmp )

