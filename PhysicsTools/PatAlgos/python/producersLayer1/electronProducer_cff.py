import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import *

from CommonTools.ParticleFlow.Isolation.pfElectronIsolation_cff import *

sourceElectrons = 'gedGsfElectrons'

patElectrons.electronSource = sourceElectrons

elPFIsoDepositCharged.src = sourceElectrons
elPFIsoDepositChargedAll.src = sourceElectrons
elPFIsoDepositNeutral.src = sourceElectrons
elPFIsoDepositGamma.src = sourceElectrons
elPFIsoDepositPU.src = sourceElectrons

patElectrons.isoDeposits = cms.PSet(
    pfChargedHadrons = cms.InputTag("elPFIsoDepositCharged" ),
    pfChargedAll = cms.InputTag("elPFIsoDepositChargedAll" ),
    pfPUChargedHadrons = cms.InputTag("elPFIsoDepositPU" ),
    pfNeutralHadrons = cms.InputTag("elPFIsoDepositNeutral" ),
    pfPhotons = cms.InputTag("elPFIsoDepositGamma" ),
    )

patElectrons.isolationValues = cms.PSet(
    pfChargedHadrons = cms.InputTag("elPFIsoValueCharged04PFId"),
    pfChargedAll = cms.InputTag("elPFIsoValueChargedAll04PFId"),
    pfPUChargedHadrons = cms.InputTag("elPFIsoValuePU04PFId" ),
    pfNeutralHadrons = cms.InputTag("elPFIsoValueNeutral04PFId" ),
    pfPhotons = cms.InputTag("elPFIsoValueGamma04PFId" ),
    )

patElectrons.isolationValuesNoPFId = cms.PSet(
    pfChargedHadrons = cms.InputTag("elPFIsoValueCharged04NoPFId"),
    pfChargedAll = cms.InputTag("elPFIsoValueChargedAll04NoPFId"),
    pfPUChargedHadrons = cms.InputTag("elPFIsoValuePU04NoPFId" ),
    pfNeutralHadrons = cms.InputTag("elPFIsoValueNeutral04NoPFId" ),
    pfPhotons = cms.InputTag("elPFIsoValueGamma04NoPFId" )
    )

## for scheduled mode
makePatElectrons = cms.Sequence(
    pfElectronIsolationSequence *
    electronMatch *
    patElectrons
    )
