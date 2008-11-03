import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorModule.dtChamberEfficiencyTask_cfi import *
from DQM.DTMonitorModule.dtSegmentTask_cfi import *
dtSegmentAnalysisMonitor.detailedAnalysis = True
from DQM.DTMonitorModule.dtResolutionTask_cfi import *

dqmInfoDT = cms.EDFilter("DQMEventInfo",
                         subSystemFolder = cms.untracked.string('DT')
                         )

dtSources = cms.Sequence(dtChamberEfficiencyMonitor*dtSegmentAnalysisMonitor*dtResolutionAnalysisMonitor*dqmInfoDT)
