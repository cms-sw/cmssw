import FWCore.ParameterSet.Config as cms

pfIsolatedMuons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfMuonsPtGt5"),
    IsoDeposit = cms.InputTag("pfMuonIsolationFromDeposits"),
    IsolationCut = cms.double(2.5),
    )
