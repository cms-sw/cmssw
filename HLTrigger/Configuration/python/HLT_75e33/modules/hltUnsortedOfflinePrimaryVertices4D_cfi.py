import FWCore.ParameterSet.Config as cms

hltUnsortedOfflinePrimaryVertices4D = cms.EDProducer("PrimaryVertexProducer",
    TkClusParameters = cms.PSet(
        TkDAClusParameters = cms.PSet(
            Tmin = cms.double(4.0),
            Tpurge = cms.double(4.0),
            Tstop = cms.double(2.0)
        ),
        algorithm = cms.string('DA2D_vect')
    ),
    TkFilterParameters = cms.PSet(
        algorithm = cms.string('filter'),
        maxD0Error = cms.double(1.0),
        maxD0Significance = cms.double(4.0),
        maxDzError = cms.double(1.0),
        maxEta = cms.double(4.0),
        maxNormalizedChi2 = cms.double(10.0),
        minPixelLayersWithHits = cms.int32(2),
        minPt = cms.double(0.0),
        minSiliconLayersWithHits = cms.int32(5),
        trackQuality = cms.string('any')
    ),
    TrackLabel = cms.InputTag("hltGeneralTracks"),
    TrackTimeResosLabel = cms.InputTag("hltTofPID4DnoPID","sigmat0safe"),
    TrackTimesLabel = cms.InputTag("hltTofPID4DnoPID","t0safe"),
    beamSpotLabel = cms.InputTag("hltOnlineBeamSpot"),
    isRecoveryIteration = cms.bool(False),
    minTrackTimeQuality = cms.double(0.8),
    recoveryVtxCollection = cms.InputTag(""),
    trackMTDTimeQualityVMapTag = cms.InputTag("hltMtdTrackQualityMVA","mtdQualMVA"),
    useMVACut = cms.bool(False),
    verbose = cms.untracked.bool(False),
    vertexCollections = cms.VPSet(
        cms.PSet(
            algorithm = cms.string('AdaptiveVertexFitter'),
            chi2cutoff = cms.double(2.5),
            label = cms.string(''),
            maxDistanceToBeam = cms.double(1.0),
            minNdof = cms.double(0.0),
            useBeamConstraint = cms.bool(False),
            vertexTimeParameters = cms.PSet(
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
                legacy4D = cms.PSet(),
                algorithm = cms.string('fromTracksPID')
            )
        ),
        cms.PSet(
            algorithm = cms.string('AdaptiveVertexFitter'),
            chi2cutoff = cms.double(2.5),
            label = cms.string('WithBS'),
            maxDistanceToBeam = cms.double(1.0),
            minNdof = cms.double(2.0),
            useBeamConstraint = cms.bool(True),
            vertexTimeParameters = cms.PSet(
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
                legacy4D = cms.PSet(),
                algorithm = cms.string('fromTracksPID')
            )
        )
    )
)
