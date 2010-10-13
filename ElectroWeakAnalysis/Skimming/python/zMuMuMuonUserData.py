import FWCore.ParameterSet.Config as cms
import copy

userDataMuons = cms.EDProducer(
    "ZMuMuMuonUserData",
    src = cms.InputTag("selectedPatMuonsTriggerMatch"),
    zGenParticlesMatch = cms.InputTag(""),
##    zGenParticlesMatch = cms.InputTag(""),
    alpha = cms.double("0.75"),
    beta = cms.double("-0.75"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    ptThreshold = cms.double("1.5"),
    etEcalThreshold = cms.double("0.2"),
    etHcalThreshold = cms.double("0.5"),
    dRVetoTrk = cms.double("0.015"),
    dRTrk = cms.double("0.3"),
    dREcal = cms.double("0.25"),
    dRHcal = cms.double("0.25"),
    hltPath = cms.string("HLT_Mu11")
    )

userDataTracks = cms.EDProducer(
    "ZMuMuTrackUserData",
    src = cms.InputTag("selectedPatTracks"),
    zGenParticlesMatch = cms.InputTag(""),
    alpha = cms.double("0.75"),
    beta = cms.double("-0.75"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    ptThreshold = cms.double("1.5"),
    etEcalThreshold = cms.double("0.2"),
    etHcalThreshold = cms.double("0.5"),
    dRVetoTrk = cms.double("0.015"),
    dRTrk = cms.double("0.3"),
    dREcal = cms.double("0.25"),
    dRHcal = cms.double("0.25"),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    )

userDataDimuons= cms.EDProducer(
    "ZMuMuUserData",
    src = cms.InputTag("dimuons"),
    zGenParticlesMatch = cms.InputTag(""),
##    zGenParticlesMatch = cms.InputTag("allDimuonsMCMatch"),
    alpha = cms.double("0.75"),
    beta = cms.double("-0.75"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    hltPath = cms.string("HLT_Mu11")
    )

userDataDimuonsOneTrack= cms.EDProducer(
    "ZMuMuUserDataOneTrack",
    src = cms.InputTag("dimuonsOneTrack"),
    zGenParticlesMatch = cms.InputTag(""),
#    zGenParticlesMatch = cms.InputTag("allDimuonsMCMatch"),
    alpha = cms.double("0.75"),
    beta = cms.double("-0.75"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    hltPath = cms.string("HLT_Mu11")
    )





