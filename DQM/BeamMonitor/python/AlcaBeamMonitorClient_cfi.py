import FWCore.ParameterSet.Config as cms

AlcaBeamMonitorClient = cms.EDAnalyzer("AlcaBeamMonitorClient",
                              	 MonitorName = cms.untracked.string('AlcaBeamMonitor'),
                               )
