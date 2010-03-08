import FWCore.ParameterSet.Config as cms


userDataMuons = cms.EDProducer(
    "ZMuMuMuonUserData",
    src = cms.InputTag("selectedPatMuonsTriggerMatch"),
    zGenParticlesMatch = cms.InputTag(""),
    #ptThreshold = cms.double("1.5"),
    #etEcalThreshold = cms.double("0.2"),
    #etHcalThreshold = cms.double("0.5"),
    #deltaRVetoTrk = cms.double("0.015"),
    #deltaRTrk = cms.double("0.3"),
    #deltaREcal = cms.double("0.25"),
    #deltaRHcal = cms.double("0.25"),
    alpha = cms.double("0."),
    beta = cms.double("-0.75"),
    #relativeIsolation = cms.bool(False)
    beamSpot = cms.InputTag("offlineBeamSpot"),
    primaryVertices = cms.InputTag("offlinePrimaryVerticesWithBS")
    
    )

