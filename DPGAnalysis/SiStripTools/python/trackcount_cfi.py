import FWCore.ParameterSet.Config as cms

trackcount = cms.EDAnalyzer('TrackCount',
                      trackCollection = cms.InputTag('ctfWithMaterialTracksP5')
                      )
