import FWCore.ParameterSet.Config as cms

dqmBeamCondMonitor = cms.EDAnalyzer("BeamConditionsMonitor",
                              monitorName = cms.untracked.string('BeamMonitor'),
                              beamSpot = cms.untracked.InputTag('offlineBeamSpot') ## hltOfflineBeamSpot for HLTMON
                              )
