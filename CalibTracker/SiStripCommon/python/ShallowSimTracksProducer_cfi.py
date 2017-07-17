import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi import *

shallowSimTracks = cms.EDProducer("ShallowSimTracksProducer",
                                  Associator=cms.InputTag('trackAssociatorByHits'),
                                  TrackingParticles=cms.InputTag("mix:MergedTrackTruth"),
                                  Tracks=cms.InputTag("generalTracks",""),
                                  Prefix=cms.string("strack"),
                                  Suffix=cms.string(""))

