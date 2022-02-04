import FWCore.ParameterSet.Config as cms

L1TkPhotonsCrystal = cms.EDProducer("L1TkEmParticleProducer",
    CHI2MAX = cms.double(100.0),
    DRmax = cms.double(0.3),
    DRmin = cms.double(0.07),
    DeltaZMax = cms.double(0.6),
    ETmin = cms.double(-1),
    IsoCut = cms.double(-0.1),
    L1EGammaInputTag = cms.InputTag("L1EGammaClusterEmuProducer"),
    L1TrackInputTag = cms.InputTag("TTTracksFromTrackletEmulation","Level1TTTracks"),
    L1VertexInputTag = cms.InputTag("L1TkPrimaryVertex"),
    PTMINTRA = cms.double(2.0),
    PrimaryVtxConstrain = cms.bool(False),
    RelativeIsolation = cms.bool(True),
    ZMAX = cms.double(25.0),
    label = cms.string('EG')
)

