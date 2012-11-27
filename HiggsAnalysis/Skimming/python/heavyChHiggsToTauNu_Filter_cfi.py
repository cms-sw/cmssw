import FWCore.ParameterSet.Config as cms

heavyChHiggsToTauNuFilter = cms.EDFilter("HeavyChHiggsToTauNuSkim",
    # Collection to be accessed
    HLTTauCollection = cms.InputTag("isolatedL3SingleTau"),
    minDRFromTau = cms.double(0.5),
    DebugHeavyChHiggsToTauNuSkim = cms.bool(False),
    jetEtaMin = cms.double(-2.5),
    jetEtaMax = cms.double(2.5),
    minNumberOfJets = cms.int32(3),
    jetEtMin = cms.double(20.0),
    JetTagCollection = cms.InputTag("iterativeCone5CaloJets")
)


