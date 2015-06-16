import FWCore.ParameterSet.Config as cms
CentralityDQM = cms.EDAnalyzer(
    "CentralityDQM",
    centralitycollection = cms.InputTag("hiCentrality"),
    vertexcollection = cms.InputTag("hiSelectedVertex"),
    eventplanecollection= cms.InputTag("hiEvtPlane")

    )
