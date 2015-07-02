import FWCore.ParameterSet.Config as cms

# comment

simTrackIdProducer = cms.EDProducer("SimTrackIdProducer",
                                    trackClassifier = cms.InputTag("","QualityMasks")
#                                    trajectories = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepClusters.trajectories,                
 #                                   TrackQuality= cms.string('highPurity'),
  #                                  maxChi2=cms.double(9)
                                )
