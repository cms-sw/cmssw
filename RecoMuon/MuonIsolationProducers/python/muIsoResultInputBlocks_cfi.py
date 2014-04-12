import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
UnweightedInputTkDeps = cms.PSet(
    DepositTag = cms.InputTag("muIsoDepositTk"),
    DepositThreshold = cms.double(-1.0),
    DepositWeight = cms.double(1.0)
)
UnweightedInputTowEcalDeps = cms.PSet(
    DepositTag = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
    DepositThreshold = cms.double(-1.0),
    DepositWeight = cms.double(1.0)
)
UnweightedInputTowHcalDeps = cms.PSet(
    DepositTag = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
    DepositThreshold = cms.double(-1.0),
    DepositWeight = cms.double(1.0)
)


