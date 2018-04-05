import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
CentralityDQM = DQMEDAnalyzer(
    "CentralityDQM",
    centralitycollection = cms.InputTag("hiCentrality"),
    centralitybincollection = cms.InputTag("centralityBin","HFtowers"),
    vertexcollection = cms.InputTag("hiSelectedVertex"),
    eventplanecollection= cms.InputTag("hiEvtPlane")
    )
