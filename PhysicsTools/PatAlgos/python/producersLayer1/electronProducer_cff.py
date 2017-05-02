import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import *

from PhysicsTools.PatAlgos.recoLayer0.pfParticleSelectionForIso_cff import *
from PhysicsTools.PatAlgos.recoLayer0.pfElectronIsolationPAT_cff import *

sourceElectrons = patElectrons.electronSource

elPFIsoDepositChargedPAT.src = sourceElectrons
elPFIsoDepositChargedAllPAT.src = sourceElectrons
elPFIsoDepositNeutralPAT.src = sourceElectrons
elPFIsoDepositGammaPAT.src = sourceElectrons
elPFIsoDepositPUPAT.src = sourceElectrons

patElectrons.isoDeposits = cms.PSet(
    pfChargedHadrons = cms.InputTag("elPFIsoDepositChargedPAT" ),
    pfChargedAll = cms.InputTag("elPFIsoDepositChargedAllPAT" ),
    pfPUChargedHadrons = cms.InputTag("elPFIsoDepositPUPAT" ),
    pfNeutralHadrons = cms.InputTag("elPFIsoDepositNeutralPAT" ),
    pfPhotons = cms.InputTag("elPFIsoDepositGammaPAT" ),
    )

patElectrons.isolationValues = cms.PSet(
    pfChargedHadrons = cms.InputTag("elPFIsoValueCharged04PFIdPAT"),
    pfChargedAll = cms.InputTag("elPFIsoValueChargedAll04PFIdPAT"),
    pfPUChargedHadrons = cms.InputTag("elPFIsoValuePU04PFIdPAT" ),
    pfNeutralHadrons = cms.InputTag("elPFIsoValueNeutral04PFIdPAT" ),
    pfPhotons = cms.InputTag("elPFIsoValueGamma04PFIdPAT" ),
    )

patElectrons.isolationValuesNoPFId = cms.PSet(
    pfChargedHadrons = cms.InputTag("elPFIsoValueCharged04NoPFIdPAT"),
    pfChargedAll = cms.InputTag("elPFIsoValueChargedAll04NoPFIdPAT"),
    pfPUChargedHadrons = cms.InputTag("elPFIsoValuePU04NoPFIdPAT" ),
    pfNeutralHadrons = cms.InputTag("elPFIsoValueNeutral04NoPFIdPAT" ),
    pfPhotons = cms.InputTag("elPFIsoValueGamma04NoPFIdPAT" )
    )

## for scheduled mode
makePatElectronsTask = cms.Task(
    pfParticleSelectionForIsoTask,
    pfElectronIsolationPATTask,
    electronMatch,
    patElectrons
    )
makePatElectrons = cms.Sequence(makePatElectronsTask)
