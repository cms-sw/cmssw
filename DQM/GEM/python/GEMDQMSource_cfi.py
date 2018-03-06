import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

GEMDQMSource = DQMEDAnalyzer("GEMDQMSource",
    recHitsInputLabel = cms.InputTag("gemRecHits", ""),

)
