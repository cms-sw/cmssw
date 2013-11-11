import FWCore.ParameterSet.Config as cms

from PhysicsTools.IsolationAlgos.tkIsoDeposits_cff import *
EcalIsolationForTracks = cms.EDProducer("IsolationProducerForTracks",
    highPtTracks = cms.InputTag("highPtTracks"),
    tracks = cms.InputTag("goodTracks"),
    isoDeps = cms.InputTag("tkIsoDepositCalByAssociatorTowers","ecal"),
    coneSize = cms.double(0.3),
    trackPtMin = cms.double(20.0)
)

HcalIsolationForTracks = cms.EDProducer("IsolationProducerForTracks",
    highPtTracks = cms.InputTag("highPtTracks"),
    tracks = cms.InputTag("goodTracks"),
    isoDeps = cms.InputTag("tkIsoDepositCalByAssociatorTowers","hcal"),
    coneSize = cms.double(0.3),
    trackPtMin = cms.double(20.0)
)

highPtTrackIsolations = cms.Sequence(tkIsoDeposits+EcalIsolationForTracks+HcalIsolationForTracks)
