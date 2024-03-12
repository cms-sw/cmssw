import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("LHE2HepMCConverter",
                           LHEEventProduct = cms.InputTag("source"),
                           LHERunInfoProduct = cms.InputTag("source")
                           )
# foo bar baz
# T3d2FM1aXIP50
