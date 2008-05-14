import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
MuonSimHits = cms.EDProducer("MuonSimHitProducer",
    MuonServiceProxy,
    # Muons
    MUONS = cms.PSet(
        # The muon simtrack's must be taken from there
        simModuleLabel = cms.string('famosSimHits'),
        MaxEta = cms.double(2.4),
        # The reconstruted tracks must be taken from there
        trackModuleLabel = cms.string('generalTracks'),
        # Simulate  only simtracks in this eta range
        MinEta = cms.double(-2.4),
        simModuleProcess = cms.string('MuonSimTracks'),
        # What is to be produced // Dummy, for now:
        ProduceL1Muons = cms.untracked.bool(False),
        # Debug level
        Debug = cms.untracked.bool(False),
        ProduceGlobalMuons = cms.untracked.bool(True),
        ProduceL3Muons = cms.untracked.bool(False)
    ),
    TRACKS = cms.PSet(
        # Set to true if the full pattern recognition was used
        # to reconstruct tracks in the tracker
        FullPatternRecognition = cms.untracked.bool(False)
    ),
    MuonTrajectoryUpdatorParameters = cms.PSet(
        MaxChi2 = cms.double(1000.0), ##25.0

        RescaleError = cms.bool(False),
        RescaleErrorFactor = cms.double(100.0),
        Granularity = cms.int32(0)
    )
)


