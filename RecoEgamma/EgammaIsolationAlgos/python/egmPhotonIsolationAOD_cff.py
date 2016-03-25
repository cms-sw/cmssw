import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfNoPileUpIso_cff import * 
from CommonTools.ParticleFlow.pfParticleSelection_cff import * 
from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationMiniAOD_cff import IsoConeDefinitions

pfNoPileUpCandidates = pfAllChargedHadrons.clone()
pfNoPileUpCandidates.pdgId.extend(pfAllNeutralHadronsAndPhotons.pdgId)

particleFlowTmpPtrs = cms.EDProducer("PFCandidateFwdPtrProducer",
src = cms.InputTag('particleFlow')
)

egmPhotonIsolationAOD = cms.EDProducer( "CITKPFIsolationSumProducer",
			  srcToIsolate = cms.InputTag("gedPhotons"),
			  srcForIsolationCone = cms.InputTag('pfNoPileUpCandidates'),
			  isolationConeDefinitions = IsoConeDefinitions
  )	

egmPhotonIsolationAODSequence = cms.Sequence(particleFlowTmpPtrs + pfParticleSelectionSequence + pfNoPileUpCandidates + egmPhotonIsolationAOD)

