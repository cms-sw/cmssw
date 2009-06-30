import FWCore.ParameterSet.Config as cms

isolatedMuons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfMuonsPtGt5"),
    IsoDeposit = cms.InputTag("muonIsolatorFromDeposits"),
    IsolationCut = cms.double(2.5),
    )
