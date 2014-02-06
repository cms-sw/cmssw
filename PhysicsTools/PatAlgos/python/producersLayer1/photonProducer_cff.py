import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import *

from PhysicsTools.PatAlgos.recoLayer0.pfParticleSelectionForIso_cff import *
from CommonTools.ParticleFlow.Isolation.pfPhotonIsolation_cff import *

sourcePhotons = patPhotons.photonSource

phPFIsoDepositCharged.src = sourcePhotons
phPFIsoDepositChargedAll.src = sourcePhotons
phPFIsoDepositNeutral.src = sourcePhotons
phPFIsoDepositGamma.src = sourcePhotons
phPFIsoDepositPU.src = sourcePhotons

patPhotons.isoDeposits = cms.PSet(
    pfChargedHadrons = cms.InputTag("phPFIsoDepositCharged" ),
    pfChargedAll = cms.InputTag("phPFIsoDepositChargedAll" ),
    pfPUChargedHadrons = cms.InputTag("phPFIsoDepositPU" ),
    pfNeutralHadrons = cms.InputTag("phPFIsoDepositNeutral" ),
    pfPhotons = cms.InputTag("phPFIsoDepositGamma" ),
    )

patPhotons.isolationValues = cms.PSet(
    pfChargedHadrons = cms.InputTag("phPFIsoValueCharged04PFId"),
    pfChargedAll = cms.InputTag("phPFIsoValueChargedAll04PFId"),
    pfPUChargedHadrons = cms.InputTag("phPFIsoValuePU04PFId" ),
    pfNeutralHadrons = cms.InputTag("phPFIsoValueNeutral04PFId" ),
    pfPhotons = cms.InputTag("phPFIsoValueGamma04PFId" ),
    )

## for scheduled mode
makePatPhotons = cms.Sequence(
    pfParticleSelectionForIsoSequence *
    pfPhotonIsolationSequence *
    photonMatch *
    patPhotons
    )
