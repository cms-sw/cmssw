# Test pseudorandom number generation.
# In this process random numbers are generated
# in the TestRandomNumberServiceAnalyzer.  The
# state of the random number engines is saved in
# two places.  The states are saved in a text file
# that gets written first after beginRun and then
# is overwritten after processing each event.
# The states are also saved into the event data
# for each event in the root data file produced
# by the PoolOutputModule.

# The analyzer also writes each generated random number
# to a text file named testRandomService.txt, which
# can be examined to determine that the expected
# sequence of numbers has been generated.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST1")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",

    # Tell the service to save the state of all engines
    # to a separate file written initially after beginJob
    # and then overwritten after processing each event
    saveFileName = cms.untracked.string('StashState.data'),

    # Next we specify a seed or seeds for each module that
    # uses random numbers.

    # Optionally, one can specify different types
    # of engines.  Currently the only three types
    # implemented are HepJamesRandom, RanecuEngine, and
    # TRandom3.  If you do not specify the engine, it
    # defaults to HepJamesRandom.

    # Use the parameter initialSeed for engines requiring
    # only one seed and initialSeedSet for the other ones.
    # HepJamesRandom requires one seed between 0 and 900000000
    # RanecuEngine requires two seeds between 0 and 2147483647

    t1 = cms.PSet(
        initialSeed = cms.untracked.uint32(81)
    ),
    t2 = cms.PSet(
        engineName = cms.untracked.string('RanecuEngine'),
        initialSeedSet = cms.untracked.vuint32(1, 2)
    ),
    t3 = cms.PSet(
        initialSeed = cms.untracked.uint32(83),
        engineName = cms.untracked.string('TRandom3')
    ),
    t4 = cms.PSet(
        initialSeed = cms.untracked.uint32(84)
    ),
    t5 = cms.PSet(
        initialSeed = cms.untracked.uint32(191),
        engineName = cms.untracked.string('TRandom3')
    ),
    # If the source needs random numbers, then it will
    # also need a seed.  EmptySource does not generate any
    # random numbers so we do not really need "theSource"
    # here, I just added to show how it is done for sources
    # that do need random numbers.
    theSource = cms.PSet(
        engineName = cms.untracked.string('RanecuEngine'),
        initialSeedSet = cms.untracked.vuint32(7, 11)
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

# The RandomNumberGeneratorService should work with
# any kind of source
process.source = cms.Source("EmptySource")

process.t1 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t2 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t3 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t4 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")

# If you do not want to save the state of the random engines
# leave this line out.
# Including this producer causes the states to be stored
# in the event.  The label used here must be referenced
# in a later process to restore the state of the engines.
process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRandomService1.root')
)

process.p = cms.Path(process.t1+process.t2+process.t3+process.t4+process.randomEngineStateProducer)
process.o = cms.EndPath(process.out)
