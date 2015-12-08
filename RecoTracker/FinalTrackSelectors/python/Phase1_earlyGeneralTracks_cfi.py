import FWCore.ParameterSet.Config as cms
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *

# FIXME: detachedTripletStep is dropped temporarily
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
earlyGeneralTracks =  TrackCollectionMerger.clone()
earlyGeneralTracks.trackProducers = ['initialStepTracks',
                                     'highPtTripletStepTracks',
                                     'jetCoreRegionalStepTracks',
                                     'lowPtQuadStepTracks',
                                     'lowPtTripletStepTracks',
                                     'detachedQuadStepTracks',
#                                     'detachedTripletStepTracks', # FIXME: disabled for now
                                     'mixedTripletStepTracks',
                                     'pixelLessStepTracks',
                                     'tobTecStepTracks'
                                     ]
earlyGeneralTracks.inputClassifiers =["initialStep",
                                      "highPtTripletStep",
                                      "jetCoreRegionalStep",
                                      "lowPtQuadStep",
                                      "lowPtTripletStep",
                                      "detachedQuadStep",
#                                      "detachedTripletStep", # FIXME: disabled for now
                                      "mixedTripletStep",
                                      "pixelLessStep",
                                      "tobTecStep"
                                      ]
