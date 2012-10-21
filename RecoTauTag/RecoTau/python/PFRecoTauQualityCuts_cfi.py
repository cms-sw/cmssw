import FWCore.ParameterSet.Config as cms

# A set of quality cuts used for the PFTaus.  Note that the quality cuts are
# different for the signal and isolation regions.  (Currently, only in Nhits)

PFTauQualityCuts = cms.PSet(
    signalQualityCuts = cms.PSet(
        minTrackPt                   = cms.double(0.5),  # filter PFChargedHadrons below given pt
        maxTrackChi2                 = cms.double(100.), # require track Chi2
        maxTransverseImpactParameter = cms.double(0.03), # w.r.t. PV
        maxDeltaZ                    = cms.double(0.2),  # w.r.t. PV
        #minTrackVertexWeight         = cms.double(10e-4), # Tracks weight in vertex
        minTrackVertexWeight         = cms.double(-1), # Tracks weight in vertex
        minTrackPixelHits            = cms.uint32(0),    # pixel-only hits (note that these cuts are turned off,
        # the tracking cuts might be higher)
        minTrackHits                 = cms.uint32(3),    # total track hits
        minGammaEt                   = cms.double(0.5),  # filter PFgammas below given Pt
        #useTracksInsteadOfPFHadrons  = cms.bool(False),  # if true, use generalTracks, instead of PFChargedHadrons
    ),
    isolationQualityCuts = cms.PSet(
        minTrackPt                   = cms.double(1.0),
        maxTrackChi2                 = cms.double(100.),
        maxTransverseImpactParameter = cms.double(0.03),
        maxDeltaZ                    = cms.double(0.2),
        # Optionally cut on DZ to lead track
        # This option only works for isolation, not signal!
        # maxDeltaZToLeadTrack         = cms.double(0.2),
        minTrackVertexWeight         = cms.double(-1), # Tracks weight in vertex
        minTrackPixelHits            = cms.uint32(0),
        minTrackHits                 = cms.uint32(8),
        minGammaEt                   = cms.double(1.5),
        #useTracksInsteadOfPFHadrons  = cms.bool(False),
    ),
    # The central definition of primary vertex source.
    primaryVertexSrc = cms.InputTag("offlinePrimaryVertices"),
    # Possible algorithms are: highestPtInEvent, closestInDeltaZ,
    # highestWeightForLeadTrack
    pvFindingAlgo = cms.string("highestWeightForLeadTrack"),
)
