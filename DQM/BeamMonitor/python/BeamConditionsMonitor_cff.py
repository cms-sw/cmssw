import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

dqmBeamCondMonitor = DQMEDAnalyzer("BeamConditionsMonitor",
                              monitorName = cms.untracked.string('BeamMonitor'),
                              beamSpot = cms.untracked.InputTag('offlineBeamSpot') ## hltOfflineBeamSpot for HLTMON
                              )
