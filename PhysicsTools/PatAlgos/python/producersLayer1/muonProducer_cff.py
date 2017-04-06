import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *

from PhysicsTools.PatAlgos.recoLayer0.pfParticleSelectionForIso_cff import *
from PhysicsTools.PatAlgos.recoLayer0.pfMuonIsolationPAT_cff import *

sourceMuons = patMuons.muonSource

muPFIsoDepositChargedPAT.src = sourceMuons
muPFIsoDepositChargedAllPAT.src = sourceMuons
muPFIsoDepositNeutralPAT.src = sourceMuons
muPFIsoDepositGammaPAT.src = sourceMuons
muPFIsoDepositPUPAT.src = sourceMuons

patMuons.isoDeposits = cms.PSet(
    pfChargedHadrons = cms.InputTag("muPFIsoDepositChargedPAT" ),
    pfChargedAll = cms.InputTag("muPFIsoDepositChargedAllPAT" ),
    pfPUChargedHadrons = cms.InputTag("muPFIsoDepositPUPAT" ),
    pfNeutralHadrons = cms.InputTag("muPFIsoDepositNeutralPAT" ),
    pfPhotons = cms.InputTag("muPFIsoDepositGammaPAT" ),
    )

patMuons.isolationValues = cms.PSet(
    pfChargedHadrons = cms.InputTag("muPFIsoValueCharged04PAT"),
    pfChargedAll = cms.InputTag("muPFIsoValueChargedAll04PAT"),
    pfPUChargedHadrons = cms.InputTag("muPFIsoValuePU04PAT" ),
    pfNeutralHadrons = cms.InputTag("muPFIsoValueNeutral04PAT" ),
    pfPhotons = cms.InputTag("muPFIsoValueGamma04PAT" ),
    )

## for scheduled mode
makePatMuonsTask = cms.Task(
    pfParticleSelectionForIsoTask,
    muonPFIsolationPATTask,
    muonMatch,
    patMuons
    )
makePatMuons = cms.Sequence(makePatMuonsTask)
