import FWCore.ParameterSet.Config as cms

pfIsolatedElectrons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfElectronsPtGt5"),
    IsoDeposit = cms.InputTag("pfElectronIsolationFromDeposits"),
    IsolationCut = cms.double(2.5),
    )
