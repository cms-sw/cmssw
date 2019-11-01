import FWCore.ParameterSet.Config as cms
import copy
#Geometry
# include used for track reconstruction 
# note that tracking is redone since we need updated hits and they 
# are not stored in the event!
from RecoParticleFlow.PFProducer.particleFlowEGamma_cfi import *
from RecoEgamma.EgammaPhotonProducers.gedPhotonSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.ootPhotonSequence_cff import *
from RecoEgamma.EgammaElectronProducers.gedGsfElectronSequence_cff import *
from RecoEgamma.EgammaElectronProducers.pfBasedElectronIso_cff import *
from RecoEgamma.EgammaIsolationAlgos.particleBasedIsoProducer_cfi import particleBasedIsolation as _particleBasedIsolation
from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationAOD_cff import egmPhotonIsolation as _egmPhotonIsolationAOD
from RecoEgamma.EgammaIsolationAlgos.egmGedGsfElectronPFIsolation_cff import egmGedGsfElectronPFNoPileUpIsolationMapBasedVeto as _egmElectronIsolationCITK
from RecoEgamma.EgammaIsolationAlgos.egmGedGsfElectronPFIsolation_cff import egmGedGsfElectronPFPileUpIsolationMapBasedVeto as _egmElectronIsolationCITKPileUp


from CommonTools.ParticleFlow.pfNoPileUpIso_cff import pfPileUpIso, pfNoPileUpIso, pfNoPileUpIsoSequence
from CommonTools.ParticleFlow.ParticleSelectors.pfSortByType_cff import pfPileUpAllChargedParticles 
from RecoEgamma.EgammaIsolationAlgos.egmIsolationDefinitions_cff import pfNoPileUpCandidates
from RecoEgamma.EgammaIsolationAlgos.egmIsoConeDefinitions_cfi import IsoConeDefinitions

particleBasedIsolationTmp = _particleBasedIsolation.clone()
particleBasedIsolationTmp.photonProducer =  cms.InputTag("gedPhotonsTmp")
particleBasedIsolationTmp.electronProducer = cms.InputTag("gedGsfElectronsTmp")
particleBasedIsolationTmp.pfCandidates = cms.InputTag("particleFlowTmp")
particleBasedIsolationTmp.valueMapPhoPFblockIso = cms.string("gedPhotonsTmp")
particleBasedIsolationTmp.valueMapElePFblockIso = cms.string("gedGsfElectronsTmp")

egmPhotonIsolationCITK = _egmPhotonIsolationAOD.clone()
egmElectronIsolationCITK = _egmElectronIsolationCITK.clone()
egmElectronIsolationPileUpCITK = _egmElectronIsolationCITKPileUp.clone()

#change particleBasedIsolation object to tmp
IsoConeDefinitionsPhotonsTmp = copy.deepcopy(IsoConeDefinitions)
for iPSet in IsoConeDefinitionsPhotonsTmp:
  iPSet.particleBasedIsolation = cms.InputTag("particleBasedIsolationTmp", "gedPhotonsTmp")

for iPSet in egmElectronIsolationCITK.isolationConeDefinitions:
  iPSet.particleBasedIsolation = cms.InputTag("particleBasedIsolationTmp", "gedGsfElectronsTmp")

for iPSet in egmElectronIsolationPileUpCITK.isolationConeDefinitions:
  iPSet.particleBasedIsolation = cms.InputTag("particleBasedIsolationTmp", "gedGsfElectronsTmp")

#photon isolation sums
egmPhotonIsolationCITK.srcToIsolate = cms.InputTag("gedPhotonsTmp")
egmPhotonIsolationCITK.srcForIsolationCone = cms.InputTag("pfNoPileUpCandidates")
egmPhotonIsolationCITK.isolationConeDefinitions = IsoConeDefinitionsPhotonsTmp
#electrons isolation sums
egmElectronIsolationCITK.srcToIsolate = cms.InputTag("gedGsfElectronsTmp")
egmElectronIsolationCITK.srcForIsolationCone = cms.InputTag("pfNoPileUpCandidates")
#electrons pileup isolation sum
egmElectronIsolationPileUpCITK.srcToIsolate = cms.InputTag("gedGsfElectronsTmp")
egmElectronIsolationPileUpCITK.srcForIsolationCone = cms.InputTag("pfPileUpAllChargedParticles")

photonIDValueMaps = cms.EDProducer(
  "PhotonIDValueMapProducer",
  ebReducedRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
  eeReducedRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
  esReducedRecHitCollection  = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
  particleBasedIsolation = cms.InputTag("particleBasedIsolationTmp","gedPhotonsTmp"),
  pfCandidates = cms.InputTag("particleFlowTmp"),
  src = cms.InputTag("gedPhotonsTmp"),
  vertices = cms.InputTag("offlinePrimaryVertices"),
  isAOD = cms.bool(True)
  )


particleFlowEGammaFullTask = cms.Task(particleFlowEGamma, gedGsfElectronTaskTmp, gedPhotonTaskTmp, ootPhotonTask)
particleFlowEGammaFull = cms.Sequence(particleFlowEGammaFullTask)
particleFlowEGammaFinalTask = cms.Task(particleBasedIsolationTmp,
                                       pfNoPileUpIsoTask,
                                       pfNoPileUpCandidates,
                                       pfPileUpAllChargedParticles,
                                       egmPhotonIsolationCITK,
                                       egmElectronIsolationCITK,
                                       egmElectronIsolationPileUpCITK,
                                       photonIDValueMaps,
                                       gedPhotonTask,
                                       gedElectronPFIsoTask)
particleFlowEGammaFinal = cms.Sequence(particleFlowEGammaFinalTask)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toReplaceWith(particleFlowEGammaFullTask, particleFlowEGammaFullTask.copyAndExclude([ootPhotonTask]))

