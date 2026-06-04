import FWCore.ParameterSet.Config as cms

hltUnsortedOfflinePrimaryVertices = cms.EDProducer("PrimaryVertexProducer",
    TkClusParameters = cms.PSet(
        TkDAClusParameters = cms.PSet(
            Tmin = cms.double(2.0),
            Tpurge = cms.double(2.0),
            Tstop = cms.double(0.5),
            coolingFactor = cms.double(0.6),
            d0CutOff = cms.double(3.0),
            dzCutOff = cms.double(3.0),
            uniquetrkweight = cms.double(0.8),
            vertexSize = cms.double(0.006),
            zmerge = cms.double(0.01)
        ),
        algorithm = cms.string('DA_vect')
    ),
    TkFilterParameters = cms.PSet(
        algorithm = cms.string('filter'),
        maxD0Significance = cms.double(4.0),
        maxEta = cms.double(4.0),
        maxNormalizedChi2 = cms.double(10.0),
        minPixelLayersWithHits = cms.int32(2),
        minPt = cms.double(0.9),
        minSiliconLayersWithHits = cms.int32(5),
        trackQuality = cms.string('any')
    ),
    TrackLabel = cms.InputTag("hltGeneralTracks"),
    beamSpotLabel = cms.InputTag("hltOnlineBeamSpot"),
    verbose = cms.untracked.bool(False),
    vertexCollections = cms.VPSet(
        cms.PSet(
            algorithm = cms.string('AdaptiveVertexFitter'),
            chi2cutoff = cms.double(2.5),
            label = cms.string(''),
            maxDistanceToBeam = cms.double(1.0),
            minNdof = cms.double(0.0),
            useBeamConstraint = cms.bool(False)
        ),
        cms.PSet(
            algorithm = cms.string('AdaptiveVertexFitter'),
            chi2cutoff = cms.double(2.5),
            label = cms.string('WithBS'),
            maxDistanceToBeam = cms.double(1.0),
            minNdof = cms.double(2.0),
            useBeamConstraint = cms.bool(True)
        )
    )
)

_mtdVertexTimeParameters = cms.PSet(
    fromTracksPID = cms.PSet(
        trackMTDTimeVMapTag = cms.InputTag('hltTrackExtenderWithMTD', 'generalTracktmtd'),
        trackMTDTimeErrorVMapTag = cms.InputTag('hltTrackExtenderWithMTD', 'generalTracksigmatmtd'),
        trackMTDTimeQualityVMapTag = cms.InputTag('hltMtdTrackQualityMVA', 'mtdQualMVA'),
        trackMTDTofPiVMapTag = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackTofPi'),
        trackMTDTofKVMapTag = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackTofK'),
        trackMTDTofPVMapTag = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackTofP'),
        trackMTDSigmaTofPiVMapTag = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackSigmaTofPi'),
        trackMTDSigmaTofKVMapTag = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackSigmaTofK'),
        trackMTDSigmaTofPVMapTag = cms.InputTag('hltTrackExtenderWithMTD', 'generalTrackSigmaTofP'),
        minTrackVtxWeight = cms.double(0.5),
        minTrackTimeQuality = cms.double(0.8),
        probPion = cms.double(0.7),
        probKaon = cms.double(0.2),
        probProton = cms.double(0.1),
        Tstart = cms.double(256),
        coolingFactor = cms.double(0.5),
        useMVAVtxTime = cms.bool(True)
    ),
    algorithm = cms.string('fromTracksPID')
)

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
mtd_at_hlt.toModify(hltUnsortedOfflinePrimaryVertices,
    vertexCollections = cms.VPSet(
        cms.PSet(
            **hltUnsortedOfflinePrimaryVertices.vertexCollections[0].parameters_(),
            vertexTimeParameters = _mtdVertexTimeParameters
        ),
        cms.PSet(
            **hltUnsortedOfflinePrimaryVertices.vertexCollections[1].parameters_(),
            vertexTimeParameters = _mtdVertexTimeParameters
        )
    )
)
