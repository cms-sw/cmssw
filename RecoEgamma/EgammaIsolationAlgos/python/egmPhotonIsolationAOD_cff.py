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
			  isolationConeDefinitions = cms.VPSet(cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('h+'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                      particleBasedIsolation = cms.InputTag("particleBasedIsolation", "gedPhotons"),
                                    ),
                              		 cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('h0'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                      particleBasedIsolation = cms.InputTag("particleBasedIsolation", "gedPhotons"),
                                    ),
                              		 cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('gamma'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                      particleBasedIsolation = cms.InputTag("particleBasedIsolation", "gedPhotons"),
                                    )
    )
  )	

egmPhotonIsolationAODSequence = cms.Sequence(particleFlowTmpPtrs + pfParticleSelectionSequence + pfNoPileUpCandidates + egmPhotonIsolationAOD)

