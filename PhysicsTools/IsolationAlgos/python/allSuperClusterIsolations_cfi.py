import FWCore.ParameterSet.Config as cms

allSuperClusterIsolations = cms.EDProducer("CandPtIsolationProducer",
    src = cms.InputTag("allSuperClusters"),
    d0Max = cms.double(1000000.0),
    dRMin = cms.double(0.015),
    dRMax = cms.double(0.2),
    elements = cms.InputTag("allTracks"),
    ptMin = cms.double(1.5),
    dzMax = cms.double(1000000.0)
)


# foo bar baz
