import FWCore.ParameterSet.Config as cms

# comment

simTrackIdProducer = cms.EDProducer("SimTrackIdProducer",
                                trackCollection = cms.InputTag("iterativeInitialTracks")                
                                )
