import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import *

from PhysicsTools.PatAlgos.recoLayer0.pfParticleSelectionForIso_cff import *
from PhysicsTools.PatAlgos.recoLayer0.pfPhotonIsolationPAT_cff import *

sourcePhotons = patPhotons.photonSource

phPFIsoDepositChargedPAT.src = sourcePhotons
phPFIsoDepositChargedAllPAT.src = sourcePhotons
phPFIsoDepositNeutralPAT.src = sourcePhotons
phPFIsoDepositGammaPAT.src = sourcePhotons
phPFIsoDepositPUPAT.src = sourcePhotons

patPhotons.isoDeposits = cms.PSet(
    pfChargedHadrons = cms.InputTag("phPFIsoDepositChargedPAT" ),
    pfChargedAll = cms.InputTag("phPFIsoDepositChargedAllPAT" ),
    pfPUChargedHadrons = cms.InputTag("phPFIsoDepositPUPAT" ),
    pfNeutralHadrons = cms.InputTag("phPFIsoDepositNeutralPAT" ),
    pfPhotons = cms.InputTag("phPFIsoDepositGammaPAT" ),
    )

patPhotons.isolationValues = cms.PSet(
    pfChargedHadrons = cms.InputTag("phPFIsoValueCharged04PFIdPAT"),
    pfChargedAll = cms.InputTag("phPFIsoValueChargedAll04PFIdPAT"),
    pfPUChargedHadrons = cms.InputTag("phPFIsoValuePU04PFIdPAT" ),
    pfNeutralHadrons = cms.InputTag("phPFIsoValueNeutral04PFIdPAT" ),
    pfPhotons = cms.InputTag("phPFIsoValueGamma04PFIdPAT" ),
    )

## for scheduled mode
makePatPhotonsTask = cms.Task(
    pfParticleSelectionForIsoTask,
    pfPhotonIsolationPATTask,
    photonMatch,
    patPhotons
    )
makePatPhotons = cms.Sequence(makePatPhotonsTask)
