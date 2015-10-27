import FWCore.ParameterSet.Config as cms

L1TMicroGMTInputProducer = cms.EDProducer('l1t::L1TMicroGMTInputProducer',
    inputFileName = cms.string("../test/test.dat")
)
