import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("LHE2HepMCConverter",
                           VertexSmearing = cms.PSet(refToPSet_ = cms.string("VertexSmearingParameters")),
                           LHEEventProduct = cms.InputTag("source"),
                           LHERunInfoProduct = cms.InputTag("source")
                           )
