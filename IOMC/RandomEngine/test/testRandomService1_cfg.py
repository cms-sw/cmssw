# Test pseudorandom number generation.
# In this process random numbers are generated
# in the TestRandomNumberServiceAnalyzer.  The
# state of the random number engines is saved in
# two places.  The states are saved in a text file
# that gets overwritten before modules process each
# event. The states are also saved into every event
# and luminosity block.

# The analyzer also writes each generated random number
# to a text file named testRandomService.txt, which
# can be examined to determine that the expected
# sequence of numbers has been generated.

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",

    # Tell the service to save the state of all engines
    # to a separate text file which is overwritten before
    # modules begin processing on each event.
    saveFileName = cms.untracked.string('StashState1.data'),

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
    # TRandom3 will take any 32 bit unsigned integer.

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
        engineName = cms.untracked.string('HepJamesRandom'),
        initialSeed = cms.untracked.uint32(84)
    ),
    t5 = cms.PSet(
        initialSeed = cms.untracked.uint32(191),
        engineName = cms.untracked.string('TRandom3')
    ),
    enableChecking = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

# The RandomNumberGeneratorService should work with
# any kind of source
process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(100),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3)
)

process.t1 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer",
                            dump = cms.untracked.bool(True))
process.t2 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t3 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t4 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")

# If you do not want to save the state of the random engines
# leave this line out.
# Including this producer causes the states to be stored
# in the event and luminosity block.  The label used here
# must be referenced in a later process to restore the state
# of the engines.
process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testRandomService1.root')
)

process.p = cms.Path(process.t1+process.t2+process.t3+process.t4+process.randomEngineStateProducer)
process.o = cms.EndPath(process.out)
