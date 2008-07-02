import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.source = cms.Source("EmptySource")

# This test uses a single module (EDProducer).
# This EDProducer contains an algorithm that is composed
# of several other algorithms, nested a few levels deep.
process.h = cms.EDProducer("HierarchicalEDProducer",
    # The EDProducer passes nest_1 to its contained alg_1 object ...
    nest_1 = cms.PSet(
        # the alg_1 object needs an integer named count ...
        count = cms.int32(87),
        # and alg_1 contains a nested alg_2 object ...
        nest_2 = cms.PSet(
            # the alg_2 object needs a string named flavor ...
            flavor = cms.string('chocolate')
        )
    ),
    # The EDProducer directly uses a double named radius ...
    radius = cms.double(2.5)
)

process.p = cms.Path(process.h)


