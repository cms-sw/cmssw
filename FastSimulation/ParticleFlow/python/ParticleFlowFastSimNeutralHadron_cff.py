import FWCore.ParameterSet.Config as cms

# Particle Flow
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
#from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
from RecoParticleFlow.PFTracking.particleFlowTrackWithDisplacedVertex_cff import *
from RecoParticleFlow.PFProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFProducer.particleFlowBlock_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cff import *
from RecoParticleFlow.PFProducer.pfElectronTranslator_cff import *
from RecoParticleFlow.PFProducer.pfPhotonTranslator_cff import *
from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cff import *
from RecoParticleFlow.PFTracking.mergedElectronSeeds_cfi import *
from FastSimulation.ParticleFlow.FSparticleFlow_cfi import *
# The following is replaced by the MVA-based 
#from RecoParticleFlow.PFProducer.pfGsfElectronCiCSelector_cff import *
from RecoParticleFlow.PFProducer.pfGsfElectronMVASelector_cff import *
from RecoParticleFlow.PFProducer.pfLinker_cff import *
from RecoParticleFlow.PFProducer.particleFlowEGamma_cff import *
particleFlow.PFCandidate = [cms.InputTag("FSparticleFlow")]

particleFlowSimParticle.sim = 'famosSimHits'

#Deactivate the recovery of dead towers since dead towers are not simulated
particleFlowRecHitHCAL.ECAL_Compensate = cms.bool(False)
#Similarly, deactivate HF cleaning for spikes
particleFlowRecHitHCAL.ShortFibre_Cut = cms.double(1E5)
particleFlowRecHitHCAL.LongFibre_Cut = cms.double(1E5)
particleFlowRecHitHCAL.LongShortFibre_Cut = cms.double(1E5)
particleFlowRecHitHCAL.ApplyLongShortDPG = cms.bool(False)
particleFlowClusterHFEM.thresh_Clean_Barrel = cms.double(1E5)
particleFlowClusterHFEM.thresh_Clean_Endcap = cms.double(1E5)
particleFlowClusterHFHAD.thresh_Clean_Barrel = cms.double(1E5)
particleFlowClusterHFHAD.thresh_Clean_Endcap = cms.double(1E5)

#particleFlowBlock.useNuclear = cms.bool(True)
#particleFlowBlock.useConversions = cms.bool(True)
#particleFlowBlock.useV0 = cms.bool(True)

#particleFlow.rejectTracks_Bad =  cms.bool(False)
#particleFlow.rejectTracks_Step45 = cms.bool(False)

#particleFlow.usePFNuclearInteractions = cms.bool(True)
#particleFlow.usePFConversions = cms.bool(True)
#particleFlow.usePFDecays = cms.bool(True)


famosParticleFlowSequence = cms.Sequence(
    caloTowersRec+
#    pfTrackElec+
    particleFlowTrackWithDisplacedVertex+
#    pfGsfElectronCiCSelectionSequence+
    pfGsfElectronMVASelectionSequence+
    particleFlowBlock+
    particleFlowEGamma+
    particleFlowTmp+
    particleFlowTmpPtrs+
    FSparticleFlow+
    pfElectronTranslatorSequence+
    pfPhotonTranslatorSequence
)

particleFlowLinks = cms.Sequence(particleFlow+particleFlowPtrs)

# PF Reco Jets and MET

from RecoJets.JetProducers.PFJetParameters_cfi import PFJetParameters
#PFJetParameters.src = cms.InputTag("FSparticleFlow") #AG
from RecoJets.Configuration.RecoPFJets_cff import *
from RecoMET.METProducers.PFMET_cfi import *
#pfMet.src = cms.InputTag("FSparticleFlow") #AG
from RecoMET.Configuration.RecoPFMET_cff import *

PFJetMet = cms.Sequence(
    recoPFJets+
    recoPFMET
)



# Tau tagging

from FastSimulation.ParticleFlow.TauTaggingFastSimNeutralHadron_cff import *









