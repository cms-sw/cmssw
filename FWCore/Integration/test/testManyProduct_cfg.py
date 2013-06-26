# This was written to benchmark some changes to
# the getByLabel function and supporting code. It makes
# a lot of getByLabel calls although it is not particularly
# realistic in the product list it uses ...
# Not intended for use as unit test because it does not
# test anything not tested elsewhere. Only useful for
# simple benchmark tests.

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("EmptySource")

process.produceInts = cms.EDProducer("ManyProductProducer",
    nProducts = cms.untracked.uint32(1000)
)

process.getInts = cms.EDAnalyzer("ManyProductAnalyzer",
    nProducts = cms.untracked.uint32(1000)
)

process.path1 = cms.Path(process.produceInts*process.getInts)
