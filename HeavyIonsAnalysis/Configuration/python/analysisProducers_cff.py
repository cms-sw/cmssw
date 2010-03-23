import FWCore.ParameterSet.Config as cms


### correlations/flow condensed track information

allTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
                           src = cms.InputTag("hiSelectedTracks"),
                           particleType = cms.string('pi+')
                           )
