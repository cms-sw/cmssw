import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.mixFastSimObjects_cfi import *
mix = cms.EDFilter("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5),
    bunchspace = cms.int32(25),
    playback = cms.untracked.bool(False),
    mixObjects = cms.PSet(
        mixSH = cms.PSet(
            mixSimHits
        ),
        mixVertices = cms.PSet(
            mixSimVertices
        ),
        mixCH = cms.PSet(
            mixCaloHits
        ),
        mixMuonTracks = cms.PSet(
            mixMuonSimTracks
        ),
        mixTracks = cms.PSet(
            mixSimTracks
        )
    )
)


