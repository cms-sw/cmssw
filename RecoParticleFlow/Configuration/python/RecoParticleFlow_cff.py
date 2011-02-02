import FWCore.ParameterSet.Config as cms


#from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
from RecoParticleFlow.PFTracking.particleFlowTrackWithDisplacedVertex_cff import *

from RecoParticleFlow.PFProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFProducer.particleFlowBlock_cff import *

from RecoParticleFlow.PFProducer.particleFlow_cff import *
from RecoParticleFlow.PFProducer.pfElectronTranslator_cff import *

particleFlowReco = cms.Sequence( particleFlowTrackWithDisplacedVertex*
                                 particleFlowBlock*
                                 particleFlow*
                                 pfElectronTranslatorSequence)

