import FWCore.ParameterSet.Config as cms

paramMuons = cms.EDProducer("ParamL3MuonProducer",
    # Set the following to false if you want to use the default CLHEP
    # random engine
    UseTRandomEngine = cms.bool(True),
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
        # What is to be produced
        ProduceL1Muons = cms.untracked.bool(True),
        # Debug level
        Debug = cms.untracked.bool(False),
        ProduceGlobalMuons = cms.untracked.bool(True),
        ProduceL3Muons = cms.untracked.bool(True)
    ),
    TRACKS = cms.PSet(
        # Set to true if the full pattern recognition was used
        # to reconstruct tracks in the tracker
        FullPatternRecognition = cms.untracked.bool(False)
    )
)


