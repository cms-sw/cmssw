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
