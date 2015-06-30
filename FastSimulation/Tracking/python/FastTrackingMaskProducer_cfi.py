import FWCore.ParameterSet.Config as cms

# comment                                                                                                                                                                                                                                                             

fastTrackingMaskProducer = cms.EDProducer("FastTrackingMaskProducer",
#                                    trajectories = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepClusters.trajectories,                                                                                                                  
 #                                   TrackQuality= cms.string('highPurity'),                                                                                                                                                                                           
                                )



