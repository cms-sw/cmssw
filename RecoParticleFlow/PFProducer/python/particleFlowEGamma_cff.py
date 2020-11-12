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

particleBasedIsolationTmp = _particleBasedIsolation.clone(
    photonProducer        = "gedPhotonsTmp",
    electronProducer      = "gedGsfElectronsTmp",
    pfCandidates          = "particleFlowTmp",
    valueMapPhoPFblockIso = "gedPhotonsTmp",
    valueMapElePFblockIso = "gedGsfElectronsTmp"
)
#change particleBasedIsolation object to tmp
IsoConeDefinitionsPhotonsTmp = copy.deepcopy(IsoConeDefinitions)
for iPSet in IsoConeDefinitionsPhotonsTmp:
  iPSet.particleBasedIsolation = "particleBasedIsolationTmp:gedPhotonsTmp"

#photon isolation sums
egmPhotonIsolationCITK = _egmPhotonIsolationAOD.clone(
    srcToIsolate        = "gedPhotonsTmp",
    srcForIsolationCone = "pfNoPileUpCandidates",
    isolationConeDefinitions = IsoConeDefinitionsPhotonsTmp
)

#electrons isolation sums
egmElectronIsolationCITK = _egmElectronIsolationCITK.clone(
    srcToIsolate        = "gedGsfElectronsTmp",
    srcForIsolationCone = "pfNoPileUpCandidates"
)

for iPSet in egmElectronIsolationCITK.isolationConeDefinitions:
  iPSet.particleBasedIsolation = "particleBasedIsolationTmp:gedGsfElectronsTmp"

#electrons pileup isolation sum
egmElectronIsolationPileUpCITK = _egmElectronIsolationCITKPileUp.clone(
    srcToIsolate        = "gedGsfElectronsTmp",
    srcForIsolationCone = "pfPileUpAllChargedParticles"
)

for iPSet in egmElectronIsolationPileUpCITK.isolationConeDefinitions:
  iPSet.particleBasedIsolation = "particleBasedIsolationTmp:gedGsfElectronsTmp"

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

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(particleFlowEGammaFullTask, particleFlowEGammaFullTask.copyAndExclude([ootPhotonTask]))

