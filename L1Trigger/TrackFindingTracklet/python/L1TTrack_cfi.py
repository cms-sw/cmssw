import FWCore.ParameterSet.Config as cms

TTTracksFromPixelDigisTracklet = cms.EDProducer("TrackFindingTrackletProducer",
  geometry = cms.untracked.string('BE5D'),
)

#process.BeamSpotFromSim = cms.EDProducer("BeamSpotFromSimProducer")
#process.TrackFindingTracklet_step = cms.Path(process.BeamSpotFromSim*process.TTTracksFromPixelDigisTracklet)

