import FWCore.ParameterSet.Config as cms

selectTracks = cms.EDFilter("TrackSelector",
    src = cms.InputTag("generalTracks"),
    cut = cms.string('pt > 0.29 & numberOfValidHits > 7 & d0 <= 3.5 & dz <= 30')
)

allTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("selectTracks"),
    particleType = cms.string('pi+')
)

#goodTracks = cms.EDFilter("CandSelector",
#    filter = cms.bool(False),
#    src = cms.InputTag("allTracks"),
#    cut = cms.string('pt > 0.29')
#)

goodTracks = cms.EDFilter("PtMinCandViewSelector",
    src = cms.InputTag("allTracks"),
    ptMin = cms.double(0.29)
)

UEAnalysisTracks = cms.Sequence(selectTracks+allTracks+goodTracks)



