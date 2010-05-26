import FWCore.ParameterSet.Config as cms

dqmBeamCondMonitor = cms.EDFilter("BeamConditionsMonitor",
                              monitorName = cms.untracked.string('BeamMonitor'),
                              beamSpot = cms.untracked.InputTag('offlineBeamSpot'), ## hltOfflineBeamSpot for HLTMON
                              Debug = cms.untracked.bool(False)
                              )
