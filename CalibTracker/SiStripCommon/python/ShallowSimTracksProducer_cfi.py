import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *

shallowSimTracks = cms.EDProducer("ShallowSimTracksProducer",
                                  Associator=cms.ESInputTag('TrackAssociatorByHits:TrackAssociatorByHits'),
                                  TrackingParticles=cms.InputTag("mix:MergedTrackTruth"),
                                  Tracks=cms.InputTag("generalTracks",""),
                                  Prefix=cms.string("strack"),
                                  Suffix=cms.string(""))

