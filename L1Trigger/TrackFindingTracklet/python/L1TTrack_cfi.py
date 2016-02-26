import FWCore.ParameterSet.Config as cms

TTTracksFromPhase2TrackerDigisTracklet = cms.EDProducer("TrackFindingTrackletProducer",
  geometry = cms.untracked.string('BE5D'),
)

#process.BeamSpotFromSim = cms.EDProducer("BeamSpotFromSimProducer")
#process.TrackFindingTracklet_step = cms.Path(process.BeamSpotFromSim*process.TTTracksFromPhase2TrackerDigisTracklet)

