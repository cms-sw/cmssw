import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *

from PhysicsTools.PatAlgos.recoLayer0.pfParticleSelectionForIso_cff import *
from CommonTools.ParticleFlow.Isolation.pfMuonIsolation_cff import *

sourceMuons = patMuons.muonSource

muPFIsoDepositCharged.src = sourceMuons
muPFIsoDepositChargedAll.src = sourceMuons
muPFIsoDepositNeutral.src = sourceMuons
muPFIsoDepositGamma.src = sourceMuons
muPFIsoDepositPU.src = sourceMuons

patMuons.isoDeposits = cms.PSet(
    pfChargedHadrons = cms.InputTag("muPFIsoDepositCharged" ),
    pfChargedAll = cms.InputTag("muPFIsoDepositChargedAll" ),
    pfPUChargedHadrons = cms.InputTag("muPFIsoDepositPU" ),
    pfNeutralHadrons = cms.InputTag("muPFIsoDepositNeutral" ),
    pfPhotons = cms.InputTag("muPFIsoDepositGamma" ),
    )

patMuons.isolationValues = cms.PSet(
    pfChargedHadrons = cms.InputTag("muPFIsoValueCharged04"),
    pfChargedAll = cms.InputTag("muPFIsoValueChargedAll04"),
    pfPUChargedHadrons = cms.InputTag("muPFIsoValuePU04" ),
    pfNeutralHadrons = cms.InputTag("muPFIsoValueNeutral04" ),
    pfPhotons = cms.InputTag("muPFIsoValueGamma04" ),
    )

## for scheduled mode
makePatMuons = cms.Sequence(
    pfParticleSelectionForIsoSequence *
    pfMuonIsolationSequence *
    muonMatch *
    patMuons
    )
