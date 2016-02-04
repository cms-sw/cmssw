import FWCore.ParameterSet.Config as cms

shallowTracks = cms.EDProducer("ShallowTracksProducer",
                               Tracks=cms.InputTag("generalTracks",""),
                               Prefix=cms.string("track"),
                               Suffix=cms.string(""))

