import FWCore.ParameterSet.Config as cms

isolatedElectrons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfElectronsPtGt5"),
    IsoDeposit = cms.InputTag("electronIsolatorFromDeposits"),
    IsolationCut = cms.double(2.5),
    )
