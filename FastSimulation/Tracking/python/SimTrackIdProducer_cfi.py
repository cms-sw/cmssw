import FWCore.ParameterSet.Config as cms

# comment

simTrackIdProducer = cms.EDProducer("SimTrackIdProducer",
#                                    trajectories = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepClusters.trajectories,                
 #                                   TrackQuality= cms.string('highPurity'),
  #                                  maxChi2=cms.double(9)
                                )
