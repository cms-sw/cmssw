import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIsolationProducers.muIsolation_cff import *
patAODMuonIsolationLabels = cms.PSet(
    associations = cms.VInputTag(cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"), cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"), cms.InputTag("muIsoDepositCalByAssociatorTowers","ho"), cms.InputTag("muIsoDepositTk"), cms.InputTag("muIsoDepositJets"))
)
patAODMuonIsolations = cms.EDFilter("MultipleIsoDepositsToValueMaps",
    patAODMuonIsolationLabels,
    collection = cms.InputTag("muons")
)

layer0MuonIsolations = cms.EDFilter("CandManyValueMapsSkimmerIsoDeposits",
    patAODMuonIsolationLabels,
    commonLabel = cms.InputTag("patAODMuonIsolations"),
    collection = cms.InputTag("allLayer0Muons"),
    backrefs = cms.InputTag("allLayer0Muons")
)

patMuonIsolation = cms.Sequence(muIsolation)
patAODMuonIsolation = cms.Sequence(patAODMuonIsolations)
patLayer0MuonIsolation = cms.Sequence(layer0MuonIsolations)

