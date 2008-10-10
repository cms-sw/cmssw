import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorModule.dtChamberEfficiencyTask_cfi import *
from DQM.DTMonitorModule.dtSegmentTask_cfi import *
dtSegmentAnalysisMonitor.detailedAnalysis = True

dqmInfoDT = cms.EDFilter("DQMEventInfo",
                         subSystemFolder = cms.untracked.string('DT')
                         )

dtSources = cms.Sequence(dtChamberEfficiencyMonitor*dtSegmentAnalysisMonitor*dqmInfoDT)
