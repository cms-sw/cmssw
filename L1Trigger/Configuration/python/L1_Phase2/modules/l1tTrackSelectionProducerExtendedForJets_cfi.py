import FWCore.ParameterSet.Config as cms

l1tTrackSelectionProducerExtendedForJets = cms.EDProducer("L1TrackSelectionProducer",
    cutSet = cms.PSet(
        absEtaMax = cms.double(2.4),
        absZ0Max = cms.double(15.0),
        nPSStubsMin = cms.int32(0),
        nStubsMin = cms.int32(4),
        promptMVAMin = cms.double(-1.0),
        ptMin = cms.double(2.0),
        reducedBendChi2Max = cms.double(999.9),
        reducedBendChi2MaxNstub4 = cms.double(2.25),
        reducedBendChi2MaxNstub5 = cms.double(3.5),
        reducedChi2RPhiMax = cms.double(999.9),
        reducedChi2RPhiMaxNstub4 = cms.double(6.0),
        reducedChi2RPhiMaxNstub5 = cms.double(35.0),
        reducedChi2RZMax = cms.double(999.9),
        reducedChi2RZMaxNstub4 = cms.double(5.0),
        reducedChi2RZMaxNstub5 = cms.double(5.0)
    ),
    debug = cms.int32(0),
    l1TracksInputTag = cms.InputTag("l1tGTTInputProducerExtended","Level1TTTracksExtendedConverted"),
    outputCollectionName = cms.string('Level1TTTracksExtendedSelected'),
    processEmulatedTracks = cms.bool(True),
    processSimulatedTracks = cms.bool(True)
)
