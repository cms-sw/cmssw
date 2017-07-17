import FWCore.ParameterSet.Config as cms
CentralityDQM = cms.EDAnalyzer(
    "CentralityDQM",
    centralitycollection = cms.InputTag("hiCentrality"),
    centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
    vertexcollection = cms.InputTag("hiSelectedVertex"),
    eventplanecollection= cms.InputTag("hiEvtPlane")
    )
