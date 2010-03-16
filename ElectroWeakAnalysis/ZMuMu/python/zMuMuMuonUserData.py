import FWCore.ParameterSet.Config as cms
import copy

userDataMuons = cms.EDProducer(
    "ZMuMuMuonUserData",
    src = cms.InputTag("selectedPatMuonsTriggerMatch"),
    zGenParticlesMatch = cms.InputTag(""),
    alpha = cms.double("0."),
    beta = cms.double("-0.75"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    primaryVertices = cms.InputTag("offlinePrimaryVerticesWithBS"),
    hltPath = cms.string("HLT_Mu9")
    )

userDataTracks = cms.EDProducer(
    "ZMuMuTrackUserData",
    src = cms.InputTag("selectedPatTracks"),
    zGenParticlesMatch = cms.InputTag(""),
    alpha = cms.double("0."),
    beta = cms.double("-0.75"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    primaryVertices = cms.InputTag("offlinePrimaryVerticesWithBS"),
    )

userDataDimuons= cms.EDProducer(
    "ZMuMuUserData",
    src = cms.InputTag("dimuons"),
    zGenParticlesMatch = cms.InputTag(""),
    alpha = cms.double("0."),
    beta = cms.double("-0.75"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    primaryVertices = cms.InputTag("offlinePrimaryVerticesWithBS"),
    hltPath = cms.string("HLT_Mu9")
    )

userDataDimuonsOneTrack= cms.EDProducer(
    "ZMuMuUserDataOneTrack",
    src = cms.InputTag("dimuonsOneTrack"),
    zGenParticlesMatch = cms.InputTag(""),
    alpha = cms.double("0."),
    beta = cms.double("-0.75"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    primaryVertices = cms.InputTag("offlinePrimaryVerticesWithBS"),
    hltPath = cms.string("HLT_Mu9")
    )





