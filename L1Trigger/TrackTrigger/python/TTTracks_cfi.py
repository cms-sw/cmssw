import FWCore.ParameterSet.Config as cms

TTTracksFromPhase2TrackerDigis = cms.EDProducer("L1TrackProducer",
                          geometry = cms.untracked.string('BE5D')
                          )

BeamSpotFromSim =cms.EDProducer("BeamSpotFromSimProducer")
