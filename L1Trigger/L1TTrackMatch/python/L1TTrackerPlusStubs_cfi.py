import FWCore.ParameterSet.Config as cms

l1StubMatchedMuons = cms.EDProducer("L1TTrackerPlusStubsProducer",
    srcStubs = cms.InputTag("simKBmtfStubs"),
    srcTracks = cms.InputTag("TTTracksFromTracklet"),

    trackMatcherSettings = cms.PSet(
        sectorsToProcess = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11),
        verbose = cms.int32(0),
        sectorSettings = cms.PSet(
            verbose = cms.int32(0),
            propagationConstants  = cms.vdouble(0.0,0.0,0.0,0.0)
        )
        
    )
)
