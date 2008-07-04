import FWCore.ParameterSet.Config as cms

ecalIsolation = cms.EDFilter("EcalIsolation",
    JetForFilter = cms.InputTag("jetCrystalsAssociator"),
    SmallCone = cms.double(0.13),
    BigCone = cms.double(0.4),
    Pisol = cms.double(5.0)
)


