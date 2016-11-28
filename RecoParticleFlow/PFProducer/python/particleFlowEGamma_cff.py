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

from CommonTools.ParticleFlow.pfNoPileUpIso_cff import pfPileUpIso, pfNoPileUpIso, pfNoPileUpIsoSequence
from RecoEgamma.EgammaIsolationAlgos.egmIsolationDefinitions_cff import pfNoPileUpCandidates
from RecoEgamma.EgammaIsolationAlgos.egmIsoConeDefinitions_cfi import IsoConeDefinitions  as IsoConeDefinitionsTmp

particleBasedIsolationTmp = _particleBasedIsolation.clone()
particleBasedIsolationTmp.photonProducer =  cms.InputTag("gedPhotonsTmp")
particleBasedIsolationTmp.electronProducer = cms.InputTag("gedGsfElectronsTmp")
particleBasedIsolationTmp.pfCandidates = cms.InputTag("particleFlowTmp")
particleBasedIsolationTmp.valueMapPhoPFblockIso = cms.string("gedPhotonsTmp")
particleBasedIsolationTmp.valueMapElePFblockIso = cms.string("gedGsfElectronsTmp")

egmPhotonIsolationCITK = _egmPhotonIsolationAOD.clone()

for iPSet in IsoConeDefinitionsTmp:
  iPSet.particleBasedIsolation = cms.InputTag("particleBasedIsolationTmp", "gedPhotonsTmp")


egmPhotonIsolationCITK.srcToIsolate = cms.InputTag("gedPhotonsTmp")
egmPhotonIsolationCITK.srcForIsolationCone = cms.InputTag("pfNoPileUpCandidates")
egmPhotonIsolationCITK.isolationConeDefinitions = IsoConeDefinitionsTmp

particleFlowEGammaFull = cms.Sequence(particleFlowEGamma*gedGsfElectronSequenceTmp*gedPhotonSequenceTmp)
particleFlowEGammaFinal = cms.Sequence(particleBasedIsolationTmp*pfNoPileUpIsoSequence*pfNoPileUpCandidates*egmPhotonIsolationCITK*gedPhotonSequence*gedElectronPFIsoSequence)
