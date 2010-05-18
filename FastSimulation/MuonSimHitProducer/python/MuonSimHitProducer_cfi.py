import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from FastSimulation.MaterialEffects.MaterialEffects_cfi import *
MuonSimHits = cms.EDProducer("MuonSimHitProducer",
    # Material effects (Multiple scattering)
    MaterialEffectsForMuonsBlock,
    # Services
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
        # What is to be produced // Dummy, for now:
        ProduceL1Muons = cms.untracked.bool(False),
        simModuleProcess = cms.string('MuonSimTracks'),
        ProduceGlobalMuons = cms.untracked.bool(True),
        ProduceL3Muons = cms.untracked.bool(False)
    ),
    Chi2EstimatorCut = cms.double(1000.0),
    TRACKS = cms.PSet(
        # Set to true if the full pattern recognition was used
        # to reconstruct tracks in the tracker
        FullPatternRecognition = cms.untracked.bool(False)
    )
)


