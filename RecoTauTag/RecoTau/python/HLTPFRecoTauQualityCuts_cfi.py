import FWCore.ParameterSet.Config as cms

# HLT specific quality cuts - referenced by HLTPFTauDiscriminatioByIsolation

# A set of quality cuts used for the PFTaus.  Note that the quality cuts are
# different for the signal and isolation regions.  (Currently, only in Nhits)

hltPFTauQualityCuts = cms.PSet(
    signalQualityCuts = cms.PSet(
        minTrackPt                   = cms.double(0.0),  # filter PFChargedHadrons below given pt
        maxTrackChi2                 = cms.double(1000.0), # require track Chi2
        minTrackPixelHits            = cms.uint32(0),    # pixel-only hits (note that these cuts are turned off,
        # the tracking cuts might be higher)
        maxDeltaZ                    = cms.double(0.2),  # Should in general be disabled at HLT (PV is sometimes missing)
        maxTransverseImpactParameter = cms.double(0.03), # Should in general be disabled at HLT (PV is sometimes missing)
        minTrackVertexWeight         = cms.double(-1),   # Should in general be disabled at HLT (PV is sometimes missing)
        minTrackHits                 = cms.uint32(3),    # total track hits
        minGammaEt                   = cms.double(0.5),  # filter PFgammas below given Pt
        useTracksInsteadOfPFHadrons  = cms.bool(False),  # if true, use generalTracks, instead of PFChargedHadrons
    ),
    isolationQualityCuts = cms.PSet(
        minTrackPt                   = cms.double(1.5),
        maxTrackChi2                 = cms.double(100.0),
        maxDeltaZ                    = cms.double(0.2),  # Should in general be disabled at HLT (PV is sometimes missing)
        maxTransverseImpactParameter = cms.double(0.03), # Should in general be disabled at HLT (PV is sometimes missing)
        minTrackVertexWeight         = cms.double(-1),   # Should in general be disabled at HLT (PV is sometimes missing)
        # Optionally cut on DZ to lead track
        # This option only works for isolation, not signal!
        maxDeltaZToLeadTrack         = cms.double(0.2),
        minTrackPixelHits            = cms.uint32(0),
        minTrackHits                 = cms.uint32(3),
        minGammaEt                   = cms.double(1.5),
        useTracksInsteadOfPFHadrons  = cms.bool(False),
    ),
    # The central definition of primary vertex source.
    primaryVertexSrc = cms.InputTag("hltPixelVertices"),
    # Possible algorithms are: highestPtInEvent, closestInDeltaZ,
    # highestWeightForLeadTrack
    pvFindingAlgo = cms.string("highestPtInEvent"),
)
