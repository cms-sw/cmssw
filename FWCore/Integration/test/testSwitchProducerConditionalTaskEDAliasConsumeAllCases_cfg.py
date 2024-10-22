import FWCore.ParameterSet.Config as cms

import sys
enableTest2 = (sys.argv[-1] != "disableTest2")
class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda accelerators: (True, -10),
                test2 = lambda accelerators: (enableTest2, -9)
            ), **kargs)

process = cms.Process("PROD1")

process.source = cms.Source("EmptySource")
if enableTest2:
    process.source.firstLuminosityBlock = cms.untracked.uint32(2)

process.maxEvents.input = 3

process.intProducer1 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(1))
process.intProducer2 = cms.EDProducer("ManyIntProducer", ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string("foo"),value=cms.int32(2))))
if enableTest2:
    process.intProducer1.throw = cms.untracked.bool(True)
else:
    process.intProducer2.throw = cms.untracked.bool(True)

process.intProducerAlias = SwitchProducerTest(
    test1 = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducer1")),
    test2 = cms.EDAlias(intProducer2 = cms.VPSet(cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string(""), toProductInstance = cms.string("")),
                                                 cms.PSet(type = cms.string("edmtestIntProduct"), fromProductInstance = cms.string("foo"), toProductInstance = cms.string("other"))))
)
process.intProducerAliasConsumer = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("intProducerAlias"))

# Test that direct dependence on SwitchProducer case-EDAlias within
# the same ConditionalTask does not cause the aliased-for EDProducer
# (in the same ConditionalTask) to become unscheduled, also when the
# consumption is registered within callWhenNewProductsRegistered().
# Note that neither case of intProducerAlias gets run by the Paths in
# this test.
process.rejectingFilter = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(-1)
)
process.allCaseGenericConsumer = cms.EDAnalyzer("GenericConsumer", eventProducts = cms.untracked.vstring(
    "intProducerAlias@test1",
    "intProducerAlias@test2"
))
process.test1Consumer = cms.EDAnalyzer("edmtest::GenericIntsAnalyzer", srcEvent = cms.untracked.VInputTag("intProducerAlias@test1"))
process.test2Consumer = cms.EDAnalyzer("edmtest::GenericIntsAnalyzer", srcEvent = cms.untracked.VInputTag("intProducerAlias@test2"))
if enableTest2:
    process.test1Consumer.inputShouldBeMissing = cms.untracked.bool(True)
    process.test2Consumer.inputShouldExist = cms.untracked.bool(True)
else:
    process.test1Consumer.inputShouldExist = cms.untracked.bool(True)
    process.test2Consumer.inputShouldBeMissing = cms.untracked.bool(True)

process.ct = cms.ConditionalTask(process.intProducerAlias, process.intProducer1, process.intProducer2)

# This path causes the chosen case of intProducerAlias to run
process.p1 = cms.Path(process.intProducerAliasConsumer, process.ct)

# This path makes the ConditionalTask system to think that all cases
# of intProducerAlias would be run, but they are not run as part of
# this Path because their consumer is behind an EDFilter that rejects
# all events
process.p2 = cms.Path(process.rejectingFilter + process.allCaseGenericConsumer, process.ct)

# This path tests that only the chosen case of intProducerAlias was run
process.p3 = cms.Path(process.test1Consumer + process.test2Consumer)
