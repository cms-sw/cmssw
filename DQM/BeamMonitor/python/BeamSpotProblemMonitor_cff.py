import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

dqmBeamSpotProblemMonitor = DQMEDAnalyzer("BeamSpotProblemMonitor",
                              monitorName = cms.untracked.string('BeamSpotProblemMonitor'),
                              DCSStatus   = cms.untracked.InputTag('scalersRawToDigi'),
                              pixelTracks = cms.untracked.InputTag('pixelTracks'),
                              scalarBSCollection =cms.untracked.InputTag('scalersRawToDigi'), 
                              nCosmicTrk = cms.untracked.int32(10),
                              OnlineMode = cms.untracked.bool(True),
                              AlarmONThreshold = cms.untracked.int32(10),
                              AlarmOFFThreshold = cms.untracked.int32(40),
                              Debug = cms.untracked.bool(False),
                              doTest= cms.untracked.bool(False)
                              )
