import FWCore.ParameterSet.Config as cms

particleFlowSKPtrs = cms.EDProducer(
    "SoftKillerProducer",
    PFCandidates = cms.InputTag("pfNoPileUpJME"),
    Rho_EtaMax = cms.double(5.0),
    rParam       = cms.double(0.4)
    )

