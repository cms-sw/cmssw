import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common

# A set of quality cuts used for the PFTaus.  Note that the quality cuts are
# different for the signal and isolation regions.  (Currently, only in Nhits)

PFTauQualityCuts = cms.PSet(
    signalQualityCuts = cms.PSet(
        minTrackPt                   = cms.double(0.5),    # filter PFChargedHadrons below given pt
        maxTrackChi2                 = cms.double(100.),   # require track Chi2
        maxTransverseImpactParameter = cms.double(0.1),    # wrt. PV
        maxDeltaZ                    = cms.double(0.4),    # wrt. PV
        maxDeltaZToLeadTrack         = cms.double(-1.),    # wrt. leading track (hightest pT track in the jet that seeds the tau reconstruction)
        #minTrackVertexWeight         = cms.double(10e-4), # Tracks weight in vertex
        minTrackVertexWeight         = cms.double(-1.),    # Tracks weight in vertex
        minTrackPixelHits            = cms.uint32(0),      # pixel-only hits
        minTrackHits                 = cms.uint32(3),      # total track hits
        minGammaEt                   = cms.double(1.0),    # filter PFgammas below given Pt
        #useTracksInsteadOfPFHadrons  = cms.bool(False),   # if true, use generalTracks, instead of PFChargedHadrons
        minNeutralHadronEt           = cms.double(30.)
    ),
    isolationQualityCuts = cms.PSet(
        minTrackPt                   = cms.double(1.0),
        maxTrackChi2                 = cms.double(100.),
        maxTransverseImpactParameter = cms.double(0.03),   # wrt. PV
        maxDeltaZ                    = cms.double(0.2),    # wrt. PV
        maxDeltaZToLeadTrack         = cms.double(-1.),    # wrt. leading track (hightest pT track in the jet that seeds the tau reconstruction)
        minTrackVertexWeight         = cms.double(-1.),    # Tracks weight in vertex
        minTrackPixelHits            = cms.uint32(0),
        minTrackHits                 = cms.uint32(8),
        minGammaEt                   = cms.double(1.5),
        #useTracksInsteadOfPFHadrons  = cms.bool(False),
    ),
    vxAssocQualityCuts = cms.PSet(
        minTrackPt                   = cms.double(0.5),    # filter PFChargedHadrons below given pt
        maxTrackChi2                 = cms.double(100.),   # require track Chi2
        maxTransverseImpactParameter = cms.double(0.1),    # wrt. PV
        minTrackVertexWeight         = cms.double(-1.),    # Tracks weight in vertex
        minTrackPixelHits            = cms.uint32(0),      # pixel-only hits
        minTrackHits                 = cms.uint32(3),      # total track hits
        minGammaEt                   = cms.double(1.0)     # filter PFgammas below given Pt
        #useTracksInsteadOfPFHadrons  = cms.bool(False),   # if true, use generalTracks, instead of PFChargedHadrons
    ),
    # The central definition of primary vertex source.
    primaryVertexSrc = cms.InputTag("offlinePrimaryVertices"),
    # Possible algorithms are: 'highestPtInEvent', 'closestInDeltaZ', 'highestWeightForLeadTrack' and 'combined'
    pvFindingAlgo = cms.string("closestInDeltaZ"),
    vertexTrackFiltering = cms.bool(False),
    recoverLeadingTrk = cms.bool(False),
    # produce histograms when running in debug mode
    # makeHisto = cms.bool(False)
    leadingTrkOrPFCandOption = cms.string("leadPFCand")
    ##leadingTrkOrPFCandOption = cms.string("leadTrack")
    ##leadingTrkOrPFCandOption = cms.string("minLeadTrackOrPFCand")
    ##leadingTrkOrPFCandOption = cms.string("firstTrack") #default behaviour until 710 (first track in the collection)
)
phase2_common.toModify(PFTauQualityCuts,
                       isolationQualityCuts = dict(
                          maxDeltaZ = 0.15,
                          maxTransverseImpactParameter = 0.05
                       ) )
                       
