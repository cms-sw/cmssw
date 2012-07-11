import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.mixFastSimObjects_cfi import *
mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(0),
    minBunch = cms.int32(0),
    bunchspace = cms.int32(25),
    checktof = cms.bool(False),                   
    playback = cms.untracked.bool(False),
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),
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
        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)


