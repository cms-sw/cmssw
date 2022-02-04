import FWCore.ParameterSet.Config as cms

L1TkElectronsEllipticMatchCrystal = cms.EDProducer("L1TkElectronTrackProducer",
    DRmax = cms.double(0.2),
    DRmin = cms.double(0.03),
    DeltaZ = cms.double(0.6),
    ETmin = cms.double(-1.0),
    IsoCut = cms.double(-0.1),
    L1EGammaInputTag = cms.InputTag("L1EGammaClusterEmuProducer"),
    L1TrackInputTag = cms.InputTag("TTTracksFromTrackletEmulation","Level1TTTracks"),
    PTMINTRA = cms.double(2.0),
    RelativeIsolation = cms.bool(True),
    TrackChi2 = cms.double(10000000000.0),
    TrackEGammaDeltaEta = cms.vdouble(0.015, 0.025, 10000000000.0),
    TrackEGammaDeltaPhi = cms.vdouble(0.07, 0.0, 0.0),
    TrackEGammaDeltaR = cms.vdouble(0.08, 0.0, 0.0),
    TrackEGammaMatchType = cms.string('EllipticalCut'),
    TrackMinPt = cms.double(10.0),
    label = cms.string('EG'),
    maxChi2IsoTracks = cms.double(10000000000.0),
    minNStubsIsoTracks = cms.int32(0),
    useClusterET = cms.bool(False),
    useTwoStubsPT = cms.bool(False)
)

