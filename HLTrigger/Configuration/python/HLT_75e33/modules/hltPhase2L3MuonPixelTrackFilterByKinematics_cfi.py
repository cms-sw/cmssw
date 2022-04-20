import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonPixelTrackFilterByKinematics = cms.EDProducer("PixelTrackFilterByKinematicsProducer",
    chi2 = cms.double(1000.0),
    nSigmaInvPtTolerance = cms.double(0.0),
    nSigmaTipMaxTolerance = cms.double(0.0),
    ptMin = cms.double(0.9),
    tipMax = cms.double(1.0)
)
