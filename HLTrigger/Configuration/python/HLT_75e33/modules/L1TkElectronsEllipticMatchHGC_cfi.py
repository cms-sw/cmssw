import FWCore.ParameterSet.Config as cms

L1TkElectronsEllipticMatchHGC = cms.EDProducer("L1TkElectronTrackProducer",
    DRmax = cms.double(0.2),
    DRmin = cms.double(0.03),
    DeltaZ = cms.double(0.6),
    ETmin = cms.double(-1.0),
    IsoCut = cms.double(-0.1),
    L1EGammaInputTag = cms.InputTag("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts"),
    L1TrackInputTag = cms.InputTag("TTTracksFromTrackletEmulation","Level1TTTracks"),
    PTMINTRA = cms.double(2.0),
    RelativeIsolation = cms.bool(True),
    TrackChi2 = cms.double(10000000000.0),
    TrackEGammaDeltaEta = cms.vdouble(0.01, 0.01, 10000000000.0),
    TrackEGammaDeltaPhi = cms.vdouble(0.07, 0.0, 0.0),
    TrackEGammaDeltaR = cms.vdouble(0.08, 0.0, 0.0),
    TrackEGammaMatchType = cms.string('EllipticalCut'),
    TrackMinPt = cms.double(10.0),
    label = cms.string('EG'),
    maxChi2IsoTracks = cms.double(100),
    minNStubsIsoTracks = cms.int32(4),
    useClusterET = cms.bool(False),
    useTwoStubsPT = cms.bool(False)
)

