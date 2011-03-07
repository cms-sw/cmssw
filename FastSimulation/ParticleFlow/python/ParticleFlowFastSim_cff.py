import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.CommonInputs_cff import *

# Particle Flow
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
#from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
from RecoParticleFlow.PFTracking.particleFlowTrackWithDisplacedVertex_cff import *
from RecoParticleFlow.PFProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFProducer.particleFlowBlock_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cff import *
from FastSimulation.ParticleFlow.FSparticleFlow_cfi import *
from RecoParticleFlow.PFProducer.pfElectronTranslator_cff import *
from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cff import *

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
    particleFlowBlock+
    particleFlow+
    FSparticleFlow+
    pfElectronTranslatorSequence    
)

# Reco Jets and MET
from RecoJets.Configuration.RecoJetsGlobal_cff import *
#from RecoJets.Configuration.JetIDProducers_cff import *
#from RecoJets.Configuration.RecoPFJets_cff import *
from RecoMET.Configuration.RecoMET_cff import *
from RecoMET.Configuration.RecoPFMET_cff import *

PFJetMet = cms.Sequence(
    recoPFJets+
    recoPFMET
)


#PFTau tagging
from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import *
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi import *
ic5JetTracksAssociatorAtVertex.tracks = 'generalTracks'
ic5PFJetTracksAssociatorAtVertex.tracks = 'generalTracks'
from RecoTauTag.Configuration.RecoTauTag_cff import *

famosTauTaggingSequence = cms.Sequence(tautagging)

from RecoTauTag.Configuration.RecoPFTauTag_cff import *

famosPFTauTaggingSequence = cms.Sequence(PFTau)
    









