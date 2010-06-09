import FWCore.ParameterSet.Config as cms

heavyChHiggsToTauNuFilter = cms.EDFilter("HeavyChHiggsToTauNuSkim",
    # Collection to be accessed
    DebugHeavyChHiggsToTauNuSkim = cms.bool(False),
    jetEtaMin = cms.double(-2.5),
    jetEtaMax = cms.double(2.5),
    minNumberOfJets = cms.int32(4),
    jetEtMin = cms.double(20.0),
    #JetTagCollection = cms.InputTag("iterativeCone5CaloJets")
    JetTagCollection = cms.InputTag("ak5CaloJets")
)
