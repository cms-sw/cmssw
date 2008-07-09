# This is an example of how to restore the random engine states
# in order to reproduce the sequence of random numbers in all the
# modules.  In this test, we use the engine states stored in the
# event data.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2")

# Use as input a file that contains the stored random engine
# states.
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:testRandomService1.root'),

    # For example, say you are interested in reproducing the process for
    # the third event without wasting time to reprocess the rest.
    # (Alternately, one could strip out this event into a separate
    # file.)
    firstRun = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(3)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",

    # This line causes the random state to be restored.
    # The value is the module label from the previous process
    # of the producer that saved the random engine state
    restoreStateLabel = cms.untracked.string('randomEngineStateProducer'),

    # COMMENT OUT OR DELETE the lines that assign seed(s) to
    # the source.  The ability to restore the state of the
    # random engine used by the source has not been implemented
    # yet.  Currently, an exception will be thrown if one
    # supplies seeds for the source in the configuration
    # file AND tries to restore the random engine states
    # in the same process.
    # theSource = cms.PSet(
    #    engineName = cms.untracked.string('RanecuEngine'),
    #    initialSeedSet = cms.untracked.vuint32(7, 11)
    # ),

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

# It is legal to have additional modules or less modules
# with or without random numbers configured, but if you do
# they simply will not get their engine state restored.  There
# may be complicated situations where you want that, although
# if you are simply reproducing what you did before it is
# not relevant.  The algorithm in the service looks for module
# labels that match and restores the random engine state for
# those, ignoring the rest.

# I've deleted the commands that produce an output file
# and store the random states in the event.
# Normally, when you restore the random engine state you
# will not need to do that.  But if anyone
# wants to do this, it is not illegal.  Just give the
# producer a different module label so you can find the saved
# state later.  If there are two objects with the same label
# in the event, it is a problem to retrieve the correct one.
