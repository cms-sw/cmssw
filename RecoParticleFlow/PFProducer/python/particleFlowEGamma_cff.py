import FWCore.ParameterSet.Config as cms

#Geometry
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
from RecoParticleFlow.PFProducer.particleFlowEGamma_cfi import *
from RecoEgamma.EgammaPhotonProducers.gedPhotonSequence_cff import *
from RecoEgamma.EgammaElectronProducers.gedGsfElectronSequence_cff import *
from RecoEgamma.EgammaElectronProducers.pfBasedElectronIso_cff import *
from RecoEgamma.EgammaIsolationAlgos.particleBasedIsoProducer_cfi import *

particleBasedIsolationTmp = particleBasedIsolation.clone()
particleBasedIsolationTmp.photonProducer =  cms.InputTag("gedPhotonsTmp")
particleBasedIsolationTmp.electronProducer = cms.InputTag("gedGsfElectronsTmp")
particleBasedIsolationTmp.pfCandidates = cms.InputTag("particleFlowTmp")
particleBasedIsolationTmp.valueMapPhoPFblockIso = cms.string("gedPhotons")
particleBasedIsolationTmp.valueMapElePFblockIso = cms.string("gedGsfElectrons")

particleFlowEGammaFull = cms.Sequence(particleFlowEGamma*gedGsfElectronSequenceTmp*gedPhotonSequenceTmp)
particleFlowEGammaFinal = cms.Sequence(particleBasedIsolationTmp*gedPhotonSequence*gedElectronPFIsoSequence)
