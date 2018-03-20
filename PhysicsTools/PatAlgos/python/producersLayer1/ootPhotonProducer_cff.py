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
patOOTPhotons.addPFClusterIso = cms.bool(False)

# MC Match
patOOTPhotons.genParticleMatch = cms.InputTag("ootPhotonMatch") ## particles source to be used for the matching

## for scheduled mode

makePatOOTPhotonsTask = cms.Task(
    ootPhotonMatch,
    patOOTPhotons
    )

makePatOOTPhotons = cms.Sequence(makePatOOTPhotonsTask)

## For legacy reprocessing
from RecoEgamma.EgammaPhotonProducers.ootPhotonSequence_cff import *
from RecoEgamma.EgammaIsolationAlgos.pfClusterIsolation_cfi import ootPhotonEcalPFClusterIsolationProducer

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toReplaceWith(makePatOOTPhotonsTask, cms.Task(
                                     ootPhotonTask,
                                     ootPhotonEcalPFClusterIsolationProducer,
                                     makePatOOTPhotonsTask.copy()
                                     ))
#the OOT are made from scratch in re-miniAOD 
#we could put the PFCluster isolation in there when we initially make them
#but decided to emulate what is done in 80X where the the isolation is only put 
#into the pat object and value maps are saved
#hence we need to have the source to be ootPhotons not ootPhotonsTmp
run2_miniAOD_80XLegacy.toModify(ootPhotonEcalPFClusterIsolationProducer,candidateProducer = cms.InputTag('ootPhotons') )

run2_miniAOD_80XLegacy.toModify(patOOTPhotons, addPFClusterIso = cms.bool(True),ecalPFClusterIsoMap = cms.InputTag("reducedEgamma", "ootPhoEcalPFClusIso"),hcalPFClusterIsoMap = cms.InputTag("") )

