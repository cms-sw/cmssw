import FWCore.ParameterSet.Config as cms


from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
#from RecoParticleFlow.PFTracking.particleFlowTrackWithDisplacedVertex_cff import *

from RecoParticleFlow.PFProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFProducer.particleFlowBlock_cff import *

from RecoParticleFlow.PFProducer.particleFlow_cff import *
from RecoParticleFlow.PFProducer.pfElectronTranslator_cff import *
from RecoParticleFlow.PFProducer.pfPhotonTranslator_cff import *
from RecoParticleFlow.PFProducer.pfGsfElectronCiCSelector_cff import *

from RecoParticleFlow.PFProducer.pfLinker_cff import * 

from CommonTools.ParticleFlow.pfParticleSelection_cff import *

particleFlowReco = cms.Sequence( particleFlowTrackWithDisplacedVertex*
                                 pfGsfElectronCiCSelectionSequence*
                                 particleFlowBlock*
                                 particleFlowTmp*
                                 pfElectronTranslatorSequence*
                                 pfPhotonTranslatorSequence*
                                 pfParticleSelectionSequence )

particleFlowLinks = cms.Sequence( particleFlow )
