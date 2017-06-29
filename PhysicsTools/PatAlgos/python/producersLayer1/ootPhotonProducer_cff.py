import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.ootPhotonMatch_cff import *
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import *

patOOTPhotons = patPhotons.clone()
patOOTPhotons.photonSource = cms.InputTag("ootPhotons")
patOOTPhotons.embedSuperCluster      = cms.bool(False) ## whether to embed in AOD externally stored supercluster
patOOTPhotons.embedSeedCluster       = cms.bool(False) ## embed in AOD externally stored the photon's seedcluster 
patOOTPhotons.embedBasicClusters     = cms.bool(False) ## embed in AOD externally stored the photon's basic clusters 
patOOTPhotons.embedPreshowerClusters = cms.bool(False) ## embed in AOD externally stored the photon's preshower clusters 
patOOTPhotons.embedRecHits           = cms.bool(False) ## embed in AOD externally stored the RecHits - can be called from the PATPhotonProducer 
    
patOOTPhotons.isoDeposits = cms.PSet()
patOOTPhotons.isolationValues = cms.PSet()

# photon ID
patOOTPhotons.addPhotonID = cms.bool(False)
patOOTPhotons.photonIDSources = cms.PSet()

# mc matching
patOOTPhotons.addGenMatch = cms.bool(True)
patOOTPhotons.embedGenMatch = cms.bool(False)

# efficiencies
patOOTPhotons.addEfficiencies = cms.bool(False)
patOOTPhotons.efficiencies    = cms.PSet()

# resolutions
patOOTPhotons.addResolutions  = cms.bool(False)
patOOTPhotons.resolutions     = cms.PSet()

# Puppi Iso
patOOTPhotons.addPuppiIsolation = cms.bool(False)

# PFClusterIso
patOOTPhotons.addPFClusterIso = cms.bool(True)
patOOTPhotons.ecalPFClusterIsoMap = cms.InputTag("reducedEgamma", "ootPhoEcalPFClusIso")
patOOTPhotons.hcalPFClusterIsoMap = cms.InputTag("reducedEgamma", "ootPhoHcalPFClusIso")

# MC Match
patOOTPhotons.genParticleMatch = cms.InputTag("ootPhotonMatch") ## particles source to be used for the matching

## for scheduled mode

makePatOOTPhotonsTask = cms.Task(
    ootPhotonMatch,
    patOOTPhotons
    )

makePatOOTPhotons = cms.Sequence(makePatOOTPhotonsTask)

## For legacy reprocessing
## Standard Sequences
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitPS_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterPS_cfi import *
## OOT Sequences
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitOOTECAL_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterOOTECALUncorrected_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterOOTECAL_cff import *
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterOOTECAL_cff import *
from RecoEgamma.EgammaPhotonProducers.ootPhotonCore_cff import *
from RecoEgamma.EgammaPhotonProducers.ootPhotons_cff import *
from RecoEgamma.EgammaIsolationAlgos.pfClusterIsolation_cfi import *

_makePatOOTPhotonsTask = makePatOOTPhotonsTask.copy()
_makePatOOTPhotonsTask.add(
    CaloTowerConstituentsMapBuilder,
    essourceEcalSev,
    ecalSeverityLevel,
    particleFlowRecHitPS,
    particleFlowClusterPS,
    particleFlowRecHitOOTECAL,
    particleFlowClusterOOTECALUncorrected,
    particleFlowClusterOOTECAL,
    particleFlowSuperClusterOOTECAL,
    ootPhotonCore,
    ootPhotons,
    ootPhotonEcalPFClusterIsolationProducer
)

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(
    patOOTPhotons, 
    hcalPFClusterIsoMap = ""
)

run2_miniAOD_80XLegacy.toReplaceWith(
    makePatOOTPhotonsTask, _makePatOOTPhotonsTask
)
