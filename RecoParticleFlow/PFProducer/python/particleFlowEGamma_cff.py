import FWCore.ParameterSet.Config as cms

#Geometry
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
from RecoParticleFlow.PFProducer.particleFlowEGamma_cfi import *
from RecoEgamma.EgammaPhotonProducers.gedPhotonSequence_cff import *
from RecoEgamma.EgammaElectronProducers.gedGsfElectronSequence_cff import *
from RecoEgamma.EgammaElectronProducers.pfBasedElectronIso_cff import *
from RecoEgamma.EgammaIsolationAlgos.particleBasedIsoProducer_cfi import particleBasedIsolation as _particleBasedIsolation
from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationAOD_cff import egmPhotonIsolationAOD as _egmPhotonIsolationAOD

particleBasedIsolationTmp = _particleBasedIsolation.clone()
particleBasedIsolationTmp.photonProducer =  cms.InputTag("gedPhotonsTmp")
particleBasedIsolationTmp.electronProducer = cms.InputTag("gedGsfElectronsTmp")
particleBasedIsolationTmp.pfCandidates = cms.InputTag("particleFlowTmp")
particleBasedIsolationTmp.valueMapPhoPFblockIso = cms.string("gedPhotonsTmp")
particleBasedIsolationTmp.valueMapElePFblockIso = cms.string("gedGsfElectronsTmp")

egmPhotonIsolationCITK = _egmPhotonIsolationAOD.clone()

isolationConeDefinitionsTmp = cms.VPSet(cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('h+'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                      particleBasedIsolation = cms.InputTag("particleBasedIsolationTmp", "gedPhotonsTmp"),
                                    ),
                              		 cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('h0'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                      particleBasedIsolation = cms.InputTag("particleBasedIsolationTmp", "gedPhotonsTmp"),
                                    ),
                              		 cms.PSet( isolationAlgo = cms.string('PhotonPFIsolationWithMapBasedVeto'),
                                      coneSize = cms.double(0.3),
                                      isolateAgainst = cms.string('gamma'),
                                      miniAODVertexCodes = cms.vuint32(2,3),
                                      vertexIndex = cms.int32(0),
                                      particleBasedIsolation = cms.InputTag("particleBasedIsolationTmp", "gedPhotonsTmp"),
                                    )
    )
egmPhotonIsolationCITK.srcToIsolate = cms.InputTag("gedPhotonsTmp")
egmPhotonIsolationCITK.srcForIsolationCone = cms.InputTag("particleFlowTmp")
egmPhotonIsolationCITK.isolationConeDefinitions = isolationConeDefinitionsTmp

particleFlowEGammaFull = cms.Sequence(particleFlowEGamma*gedGsfElectronSequenceTmp*gedPhotonSequenceTmp)
particleFlowEGammaFinal = cms.Sequence(particleBasedIsolationTmp*egmPhotonIsolationCITK*gedPhotonSequence*gedElectronPFIsoSequence)
