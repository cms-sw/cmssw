import FWCore.ParameterSet.Config as cms


# The configuration in this file will be dumped by the DumpDependencyGraph
# service into a .json file. The json will then be read by
# test_dumpDependencyGraph_check.py, which will check that its contents match
# what this file declares. Changes to this file should be manually reflected
# into test_dumpDependencyGraph_check.py .


# It will be checked that this is the process name
process = cms.Process("DEPGRAPH")


process.DumpDependencyGraph = cms.Service(
    "DumpDependencyGraph",
    fileName=cms.untracked.string("test_dumpDependencyGraph.json"),
)

# It will be checked that process.source's type is "Source"
process.source = cms.Source("IntSource")
process.maxEvents.input = 0
process.options.numberOfThreads = 1

# It will be checked that process.first has no dependencies
process.first = cms.EDProducer("IntProducer", ivalue=cms.int32(1))

# It will be checked that process.second's class is "AddIntsProducer"
# It will be checked that process.second consumes process.first
# It will be checked that process.second has no non-event dependency
process.second = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("first"))
)

process.aliasOfSecond = cms.EDAlias(
    second=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct")))
)

# It will be checked that process.viaAlias's dependency resolves (through
# aliasOfSecond) to process.second
process.viaAlias = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfSecond"))
)

# It will be checked that process.fromSource consumes process.source
process.fromSource = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("source"))
)
process.aliasOfSource = cms.EDAlias(
    source=cms.VPSet(cms.PSet(type=cms.string("edmtestIntProduct")))
)

# It will be checked that process.viaSourceAlias, through aliasOfSource, also
# consumes to process.source
process.viaSourceAlias = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("aliasOfSource"))
)

# It will be checked that process.runConsumer's dependency on
# process.runProducer is a non-event one
process.runProducer = cms.EDProducer("NonEventIntProducer", ivalue=cms.int32(1))
process.runConsumer = cms.EDProducer(
    "NonEventIntProducer",
    ivalue=cms.int32(2),
    consumesBeginRun=cms.InputTag("runProducer", "beginRun"),
)

# It will be checked that process.unknownLabel's dependency (on a modules that
# does not exist) is reported as unresolved, not as a normal edge It will be
# also checked that no other module in this configuration has an unresolved
# dependency
process.unknownLabel = cms.EDProducer(
    "AddIntsProducer", labels=cms.VInputTag(cms.InputTag("noSuchModule"))
)

# It will be checked that process.gate's type is "EDFilter"
process.gate = cms.EDFilter(
    "ModuloEventIDFilter", modulo=cms.uint32(2), offset=cms.uint32(0)
)
process.behindGate = cms.EDProducer(
    "AddIntsProducer",
    labels=cms.VInputTag(cms.InputTag("viaAlias"), cms.InputTag("unscheduled")),
)

# It will be checked that process.unscheduled is listed, but on no path
process.unscheduled = cms.EDProducer("IntProducer", ivalue=cms.int32(2))
process.onDemand = cms.Task(process.unscheduled)

process.watcher = cms.EDAnalyzer(
    "IntTestAnalyzer",
    moduleLabel=cms.untracked.InputTag("first"),
    valueMustMatch=cms.untracked.int32(1),
)

# It will be checked that the schedule order of p, q, and e matches this:
process.p = cms.Path(
    process.first
    + process.second
    + process.viaAlias
    + process.gate
    + process.behindGate,
    process.onDemand,
)
process.q = cms.Path(
    process.fromSource
    + process.viaSourceAlias
    + process.runProducer
    + process.runConsumer
    + process.unknownLabel
)
process.e = cms.EndPath(process.watcher)

process.schedule = cms.Schedule(process.p, process.q, process.e)
