import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltInitialStepTrajectorySeedsLSTTracks = cms.EDProducer(
    "TrackFromSeedProducer",
    src = cms.InputTag("hltInitialStepTrajectorySeedsLST"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    TTRHBuilder = cms.string("hltESPTTRHBuilderWithoutRefit")
)
