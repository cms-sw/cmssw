import FWCore.ParameterSet.Config as cms

apvcyclephasemonitor = cms.EDAnalyzer('APVCyclePhaseMonitor',
                                      apvCyclePhaseCollection = cms.InputTag("APVPhases"),
)	
