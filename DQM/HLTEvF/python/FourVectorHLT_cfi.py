import FWCore.ParameterSet.Config as cms
from DQM.HLTEvF.listOfFilters_cff import filters as _filters
#from DQM.HLTEvF.listOfAllGRunFilters_cff import filters as _filters

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hltResults = DQMEDAnalyzer("FourVectorHLT",
     plotAll = cms.untracked.bool(False),
     ptMax = cms.untracked.double(100.0),
     ptMin = cms.untracked.double(0.0),
     filters = _filters,
     triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD::HLT")
)


