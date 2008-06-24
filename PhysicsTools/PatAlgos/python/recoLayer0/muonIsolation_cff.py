import FWCore.ParameterSet.Config as cms

# import Muon POG isolation config
from RecoMuon.MuonIsolationProducers.muIsolation_cff import *
# sequence to re-run muon Isolation (usually not necessary)
patMuonIsolation = cms.Sequence(muIsolation)

# define module labels for old (tk-based isodeposit) POG isolation
patAODMuonIsolationLabels = cms.VInputTag(
        cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
        cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
        cms.InputTag("muIsoDepositCalByAssociatorTowers","ho"),
        cms.InputTag("muIsoDepositTk"),
        cms.InputTag("muIsoDepositJets")
)

# read and convert to ValueMap<IsoDeposit> keyed to Candidate
patAODMuonIsolations = cms.EDFilter("MultipleIsoDepositsToValueMaps",
    collection   = cms.InputTag("muons"),
    associations = patAODMuonIsolationLabels,
)

# re-key to the candidates
layer0MuonIsolations = cms.EDFilter("CandManyValueMapsSkimmerIsoDeposits",
    collection   = cms.InputTag("allLayer0Muons"),
    backrefs     = cms.InputTag("allLayer0Muons"),
    commonLabel  = cms.InputTag("patAODMuonIsolations"),
    associations = patAODMuonIsolationLabels,
)

# sequence to run on AOD before PAT
patAODMuonIsolation = cms.Sequence(patAODMuonIsolations)

# sequence to run at the end of Layer 0
patLayer0MuonIsolation = cms.Sequence(layer0MuonIsolations)

