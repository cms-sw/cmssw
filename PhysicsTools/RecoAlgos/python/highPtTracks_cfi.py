import FWCore.ParameterSet.Config as cms

highPtTracks = cms.EDFilter("PtMinCandViewSelector",
    src = cms.InputTag("goodTracks"),
    ptMin = cms.double(20)
)


