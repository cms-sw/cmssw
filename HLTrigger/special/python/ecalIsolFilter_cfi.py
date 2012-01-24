import FWCore.ParameterSet.Config as cms

#HLTEcalIsolationFilter configuration
ecalIsolFilter = cms.EDFilter("HLTEcalIsolationFilter",
    MaxNhitInnerCone = cms.int32(1000),
    MaxNhitOuterCone = cms.int32(0),
    EcalIsolatedParticleSource = cms.InputTag("ecalIsolPartProd"),
    MaxEnergyOuterCone = cms.double(10000.0),
    MaxEtaCandidate = cms.double(1.3),
    MaxEnergyInnerCone = cms.double(10000.0),
    saveTags = cms.bool( False )
)


