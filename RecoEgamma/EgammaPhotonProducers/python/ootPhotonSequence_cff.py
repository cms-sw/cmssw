import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitOOTECAL_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterOOTECALUncorrected_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterOOTECAL_cff import *
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterOOTECAL_cff import *
from RecoEgamma.EgammaPhotonProducers.ootPhotonCore_cff import *
from RecoEgamma.EgammaPhotonProducers.ootPhotons_cff import *

# task+sequence to make OOT photons from clusters in ECAL from full PFRecHits w/o timing cut
ootPhotonTask = cms.Task(
    particleFlowRecHitOOTECAL,
    particleFlowClusterOOTECALUncorrected,
    particleFlowClusterOOTECAL, 
    particleFlowSuperClusterOOTECAL, 
    ootPhotonCore, 
    ootPhotons
    )

ootPhotonSequence = cms.Sequence(ootPhotonTask)

## For legacy reprocessing: need additional products
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import essourceEcalSev, ecalSeverityLevel
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitPS_cfi import particleFlowRecHitPS
from RecoParticleFlow.PFClusterProducer.particleFlowClusterPS_cfi import particleFlowClusterPS

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toReplaceWith(ootPhotonTask, cms.Task(
                                     CaloTowerConstituentsMapBuilder,
                                     essourceEcalSev,
                                     ecalSeverityLevel,
                                     particleFlowRecHitPS,
                                     particleFlowClusterPS,
                                     ootPhotonTask.copy()
                                     ))
