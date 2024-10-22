import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10000000

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource")

# This should run because it's consumed directly by process.consumer
process.thing = cms.EDProducer("ThingProducer")

# This should not run, because it's mot consumed by any other module
process.notRunningThing = cms.EDProducer("ThingProducer")

# This should run because it's consumed indirectly by process.consumer, via process.otherThing
process.anotherThing = cms.EDProducer("ThingProducer")

# This should run because it's consumed directly by process.consumer
process.otherThing = cms.EDProducer("OtherThingProducer",
    thingTag = cms.InputTag('anotherThing'),
    transient = cms.untracked.bool(True)
)

# Make the various modules available for unscheduled execution
process.task = cms.Task(
    process.thing,
    process.anotherThing,
    process.otherThing,
    process.notRunningThing
)

# Consumes the products of process.thing and process.otherThing, causing them to run
process.consumer = cms.EDAnalyzer("GenericConsumer",
    eventProducts = cms.untracked.vstring("*_thing_*_*", "otherThing"),
    verbose = cms.untracked.bool(True)
)

# Explicilty schedule process.consumer, causing it to run along with its dependencies, provided by process.task
process.path = cms.Path(process.consumer, process.task)

# Print the summary of all modules that were run 
# The content of the summary is tested by testGenericConsumer.sh
process.options.wantSummary = True
