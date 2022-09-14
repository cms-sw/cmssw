import FWCore.ParameterSet.Config as cms

l1tTkPrimaryVertex = cms.EDProducer("L1TkFastVertexProducer",
    CHI2MAX = cms.double(100.0),
    GenParticleInputTag = cms.InputTag("genParticles"),
    HepMCInputTag = cms.InputTag("generator"),
    HighPtTracks = cms.int32(0),
    L1TrackInputTag = cms.InputTag("l1tTTTracksFromTrackletEmulation","Level1TTTracks"),
    MonteCarloVertex = cms.bool(False),
    PTMAX = cms.double(50.0),
    PTMINTRA = cms.double(2.0),
    WEIGHT = cms.int32(1),
    ZMAX = cms.double(25.0),
    doPtComp = cms.bool(True),
    doTightChi2 = cms.bool(False),
    nBinning = cms.int32(601),
    nStubsPSmin = cms.int32(3),
    nStubsmin = cms.int32(4),
    nVtx = cms.int32(1)
)
