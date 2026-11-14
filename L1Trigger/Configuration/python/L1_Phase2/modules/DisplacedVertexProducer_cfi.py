import FWCore.ParameterSet.Config as cms

DisplacedVertexProducer = cms.EDProducer("DisplacedVertexProducer",
    cutSet = cms.PSet(
        RTMax = cms.double(20.0),
        RTMin = cms.double(0.02152),
        barrelD0Min = cms.double(0.06),
        chi2rzMax = cms.double(3.0),
        diskD0Min = cms.double(0.08),
        diskEtaMin = cms.double(0.95),
        dispD0Min = cms.double(1.0),
        etaMax = cms.double(2.4),
        overlapEtaMax = cms.double(1.7),
        overlapEtaMin = cms.double(1.1),
        overlapNStubsMin = cms.int32(4),
        promptMVADispTrackMin = cms.double(0.5),
        promptMVAMin = cms.double(0.2),
        ptMin = cms.double(3.0)
    ),
    l1TrackVertexCollectionName = cms.string('dispVertices'),
    l1TracksGTTInputTag = cms.InputTag("l1tGTTInputProducerExtended","Level1TTTracksExtendedConverted"),
    l1TracksInputTag = cms.InputTag("l1tTTTracksFromExtendedTrackletEmulation","Level1TTTracks"),
    mcTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigisExtended","Level1TTTracks"),
    model = cms.FileInPath('L1Trigger/L1TTrackMatch/data/dispVertTaggerEmulationConiferShifted13p8.json'),
    runEmulation = cms.bool(True)
)
