import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing()
options.register("includeAliasToFoo", 1,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "Include case Foo in EDAlias")
options.register("includeAliasToBar", 1,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "Include case Bar in EDAlias")
options.register("consumerGets", 1,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "Consumer gets event product")
options.register("explicitProcessName", 0,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "Use explicit process name in consumer InputTag")
options.parseArguments()

process.maxEvents.input = 1
process.source = cms.Source("EmptySource")


process.simpleProducerFoo = cms.EDProducer("OVSimpleProducer", size = cms.int32(10))
process.simpleProducerBar = cms.EDProducer("OVSimpleProducer", size = cms.int32(5))

process.simpleProducer = cms.EDAlias()
if options.includeAliasToFoo != 0:
    process.simpleProducer.simpleProducerFoo = cms.VPSet(
        cms.PSet(
            type = cms.string("edmtestSimplesOwned")
        )
    )
if options.includeAliasToBar != 0:
    process.simpleProducer.simpleProducerBar = cms.VPSet(
        cms.PSet(
            type = cms.string("edmtestSimpleDerivedsOwned"),
            fromProductInstance = cms.string("derived"),
            toProductInstance = cms.string("")
        )
    )

process.simpleViewConsumer = cms.EDAnalyzer("SimpleViewAnalyzer",
    label = cms.untracked.InputTag("simpleProducer"),
    sizeMustMatch = cms.untracked.uint32(10),
    checkSize = cms.untracked.bool(options.consumerGets != 0)
)
if options.includeAliasToFoo == 0:
    process.simpleViewConsumer.sizeMustMatch = 5
if options.explicitProcessName != 0:
    process.simpleViewConsumer.label.setProcessName("TEST")

dependsOn = []
if options.includeAliasToFoo != 0:
    dependsOn.append("simpleProducerFoo")
if options.includeAliasToBar != 0:
    dependsOn.append("simpleProducerBar")
process.PathsAndConsumesOfModulesTestService = cms.Service("PathsAndConsumesOfModulesTestService",
    modulesAndConsumes = cms.VPSet(
        cms.PSet(
            key = cms.string("simpleViewConsumer"),
            value = cms.vstring(dependsOn)
        ),
    )
)

process.t = cms.Task(
    process.simpleProducerFoo,
    process.simpleProducerBar
)

process.p = cms.Path(
    process.simpleViewConsumer,
    process.t
)
