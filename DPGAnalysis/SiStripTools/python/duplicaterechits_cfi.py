import FWCore.ParameterSet.Config as cms

duplicaterechits = cms.EDAnalyzer('DuplicateRecHits',
                                  trackCollection = cms.InputTag('generalTracks'),
                                  TTRHBuilder = cms.string('WithTrackAngle')
                      )
