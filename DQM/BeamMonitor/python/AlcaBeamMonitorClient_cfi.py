import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

AlcaBeamMonitorClient = DQMEDAnalyzer("AlcaBeamMonitorClient",
                              	 MonitorName = cms.untracked.string('AlcaBeamMonitor'),
                               )
