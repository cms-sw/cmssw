import FWCore.ParameterSet.Config as cms

pfElectrons = cms.EDProducer("PFIsolation",
    verbose = cms.untracked.bool(False),
    isolation_InnerCone_DeltaR = cms.double(1e-05),
    PFCandidates = cms.InputTag("pfElectronsPtGt5"),
    PFCandidatesForIsolation = cms.InputTag("particleFlow"),
    isolation_Cone_DeltaR = cms.double(0.2),
    max_ptFraction_InCone = cms.double(1.0)
)



