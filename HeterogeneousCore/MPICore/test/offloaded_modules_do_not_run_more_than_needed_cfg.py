import FWCore.ParameterSet.Config as cms

# Used by testOffloadedModulesDoNotRunMoreThanNeeded.sh, which offloads
# producerB, producerC, producerD and producerE to a remote process.
#
# Offloaded modules should be grouped based on when they run, not only on their
# dependencies. The configuration has two paths:
#
#    - path1: producerA -- producerB -- producerC -- producerD
#    - path2: eventIDFilter -- producerE -- producerF
#
# Each producer consumes the one before it, and producerE consumes producerD.
# eventIDFilter passes only the events with an even event ID (10 out of 20).
#
# What should happen after the split:
#    - producerA stays local, and runs on every event. Its products are sent to
#      the remote
#    - producerB, producerC and producerD form one remote group, and run on
#      every event (20 times).
#    - producerE forms a second remote group, gated by eventIDFilter, and runs
#      only on the events that pass the filter (10 times).
#    - producerE is sent back, and producerF runs locally 10 times.
#
# What should not happen:
#    - producerE should not be grouped with producerB, producerC and producerD,
#      where it would run on every event (20 times).

process = cms.Process("testOffloadedModulesDoNotRunMoreThanNeeded")

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.wantSummary = True

process.source = cms.Source("EmptySource")
process.maxEvents.input = 20

# local
process.producerA = cms.EDProducer("IntProducer", ivalue=cms.int32(1))

# offloaded, first group
process.producerB = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerA")))
process.producerC = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerB")))
process.producerD = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerC")))

# local
process.eventIDFilter = cms.EDFilter("ModuloEventIDFilter",
    modulo=cms.uint32(2), offset=cms.uint32(0))

# offloaded, second group
process.producerE = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerD")))

# local
process.producerF = cms.EDProducer("AddIntsProducer", labels=cms.VInputTag(cms.InputTag("producerE")))

process.path1 = cms.Path(process.producerA + process.producerB + process.producerC + process.producerD)
process.path2 = cms.Path(process.eventIDFilter + process.producerE + process.producerF)

process.schedule = cms.Schedule(process.path1, process.path2)
