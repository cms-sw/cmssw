# Same as testRandomService1_cfg.py except for
# different seeds, luminosity block numbers,
# event numbers, output file name, and filename
# for the text file saving the states. This creates
# the second in a sequence of 3 files that will get
# merged together
import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",

    # Tell the service to save the state of all engines
    # to a separate text file which is overwritten before
    # modules begin processing on each event.
    saveFileName = cms.untracked.string('StashState2.data'),

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
        initialSeed = cms.untracked.uint32(201)
    ),
    t2 = cms.PSet(
        engineName = cms.untracked.string('RanecuEngine'),
        initialSeedSet = cms.untracked.vuint32(202, 212)
    ),
    t3 = cms.PSet(
        initialSeed = cms.untracked.uint32(203),
        engineName = cms.untracked.string('TRandom3')
    ),
    t4 = cms.PSet(
        engineName = cms.untracked.string('HepJamesRandom'),
        initialSeed = cms.untracked.uint32(204)
    ),
    t5 = cms.PSet(
        initialSeed = cms.untracked.uint32(205),
        engineName = cms.untracked.string('TRandom3')
    ),
    enableChecking = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(3),
    firstEvent = cms.untracked.uint32(6),
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
    fileName = cms.untracked.string('testRandomService2.root')
)

process.p = cms.Path(process.t1+process.t2+process.t3+process.t4+process.randomEngineStateProducer)
process.o = cms.EndPath(process.out)
