# This is an example of how to restore the random engine states
# stored after the final event that was successfully processed
# completed its processing.
# This can be used to either reproduce exactly the processing
# of an event where the process died or continue a sequence of
# events by continuing the random number sequences used in the
# previous processes as if the process had not stopped.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# At least in the current version (release 2_1_0), this
# method of restoring the state of the random engines does
# not work if the source uses a random engine.  Any source
# that does not use random numbers should work fine.
process.source = cms.Source("EmptySource")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",

    # Tell the service to restore engine states from the
    # file written by the original process
    restoreFileName = cms.untracked.string('StashState.data'),


    # Normally the rest of the parameters for the service will
    # be identical to the previous process (here they are copied
    # from testRandomService1_cfg.py)
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
    )
)

process.t1 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t2 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t3 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")
process.t4 = cms.EDAnalyzer("TestRandomNumberServiceAnalyzer")

process.p = cms.Path(process.t1+process.t2+process.t3+process.t4)
