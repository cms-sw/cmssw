import FWCore.ParameterSet.Config as cms

l1tTrackSelectionProducer = cms.EDProducer("L1TrackSelectionProducer",
    cutSet = cms.PSet(
        absEtaMax = cms.double(2.4),
        absZ0Max = cms.double(15.0),
        nPSStubsMin = cms.int32(0),
        nStubsMin = cms.int32(4),
        promptMVAMin = cms.double(-1.0),
        ptMin = cms.double(2.0),
        reducedBendChi2Max = cms.double(2.25),
        reducedBendChi2MaxNstub4 = cms.double(999.9),
        reducedBendChi2MaxNstub5 = cms.double(999.9),
        reducedChi2RPhiMax = cms.double(20.0),
        reducedChi2RPhiMaxNstub4 = cms.double(999.9),
        reducedChi2RPhiMaxNstub5 = cms.double(999.9),
        reducedChi2RZMax = cms.double(5.0),
        reducedChi2RZMaxNstub4 = cms.double(999.9),
        reducedChi2RZMaxNstub5 = cms.double(999.9)
    ),
    debug = cms.int32(0),
    l1TracksInputTag = cms.InputTag("l1tGTTInputProducer","Level1TTTracksConverted"),
    outputCollectionName = cms.string('Level1TTTracksSelected'),
    processEmulatedTracks = cms.bool(True),
    processSimulatedTracks = cms.bool(True)
)
