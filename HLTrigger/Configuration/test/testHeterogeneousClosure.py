#! /usr/bin/env python3
"""Unit tests for HLTrigger.Configuration.heterogeneousClosure.

Everything except the cmsRun job that produces the Tracer log is exercised here,
on a Process and a Tracer log built by hand.  The tests therefore need neither
conditions, nor an input file, nor a real menu.

The Process below plays the part of a menu: an @alpaka chain fed by a legacy
chain that starts from a product of the input file, a downstream module that
nothing consumes, an unrelated chain, the fixed HLT structure the reduction wraps
the seeds in, and a Path whose status is consumed by the L1 accept filter.
"""

import os
import tempfile
import unittest

import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.heterogeneousClosure import (FrameworkGraph,
                                                          ModuleGraph,
                                                          collectInputTags,
                                                          parseTracerLog,
                                                          reduceToHeterogeneous,
                                                          scheduledLabels)

TRACER_LOG = """\
some preamble the parser has to ignore
All modules and modules in the current process whose products they consume:
(This does not include modules from previous processes or the source)
  DigiProducer/'digis'
    consumes products from these modules:
      RawProducer/'rawData'
  ClusterProducer/'soaClusters'
    consumes products from these modules:
      DigiProducer/'digis'
    consumes products during Event from these EventSetup modules:
      GeometryESProducer/'geometry' ESProducer
      UnlabelledESProducer/'' ESProducer
  L1AlgoProducer/'l1tAlgoBlock'
    consumes products from these modules:
      PathStatusInserter/'pSeed'
All EventSetup modules:
  GeometryESProducer/'geometry'
    consumes products from these EventSetup modules:
      TopologyESProducer/'topology'
  UnlabelledESProducer/'UnlabelledESProducer'
  TopologyESProducer/'topology'
  UnusedESProducer/'unused'
All modules (listed by class and label) and all their consumed products.
  ClusterProducer/'soaClusters'
    consumes:
      a line the parser has to ignore
"""


def buildProcess():
    process = cms.Process("TEST")
    process.source = cms.Source("EmptySource")

    # the fixed HLT structure; providing it here keeps the test from importing
    # the L1 accept filter out of a real menu
    process.hltTriggerType = cms.EDFilter("HLTTriggerTypeFilter")
    process.hltOnlineBeamSpot = cms.EDProducer("BeamSpotProducer")
    process.HLTBeginSequence = cms.Sequence(process.hltTriggerType +
                                            process.hltOnlineBeamSpot)
    process.hltBoolEnd = cms.EDFilter("HLTBool")
    process.HLTEndSequence = cms.Sequence(process.hltBoolEnd)
    process.hltL1GTAcceptFilter = cms.EDFilter(
        "L1GTAcceptFilter", algoBlocksTag=cms.InputTag("l1tAlgoBlock"))
    process.hltTriggerSummary = cms.EDProducer("TriggerSummary",
                                               src=cms.InputTag("unrelated"))
    process.hltTrigReport = cms.EDAnalyzer("TrigReport")
    process.HLTriggerFinalPath = cms.Path(process.hltTriggerSummary)
    process.HLTAnalyzerEndpath = cms.EndPath(process.hltTrigReport)

    # rawData is defined but never scheduled: it stands for a product that the
    # input file already holds, and the closure has to stop there
    process.rawData = cms.EDProducer("RawProducer")
    process.digis = cms.EDProducer("DigiProducer", src=cms.InputTag("rawData"))
    process.soaClusters = cms.EDProducer("ClusterProducer@alpaka",
                                         src=cms.InputTag("digis"))
    process.soaHits = cms.EDProducer("HitProducer@alpaka",
                                     src=cms.InputTag("soaClusters"))
    # downstream of the seeds, so outside the closure
    process.legacyHits = cms.EDProducer("LegacyHits", src=cms.InputTag("soaHits"))
    process.unrelated = cms.EDProducer("Unrelated")
    # an @alpaka module the configuration does not schedule, so not a seed
    process.offClusters = cms.EDProducer("OffProducer@alpaka")

    process.l1Seed = cms.EDFilter("L1Seed")
    process.l1tAlgoBlock = cms.EDProducer("L1AlgoProducer")

    process.geometry = cms.ESProducer("GeometryESProducer")
    process.topology = cms.ESProducer("TopologyESProducer")
    process.unused = cms.ESProducer("UnusedESProducer")
    process.alpakaES = cms.ESProducer("SomeESProducer@alpaka")
    process.emptySource = cms.ESSource("EmptyESSource")

    process.pSeed = cms.Path(process.l1Seed)
    process.pAlgo = cms.Path(process.l1tAlgoBlock)
    process.reco = cms.Path(process.digis + process.soaClusters +
                            process.soaHits + process.legacyHits)
    process.other = cms.Path(process.unrelated)
    process.schedule = cms.Schedule(process.pSeed, process.pAlgo, process.reco,
                                    process.other, process.HLTriggerFinalPath,
                                    process.HLTAnalyzerEndpath)
    return process


# what the Tracer would report for the Process above
FRAMEWORK = FrameworkGraph(edToEd={"l1tAlgoBlock": set(["pSeed"])},
                           edToEs={"soaClusters": set(["geometry"]),
                                   "soaHits": set(["alpakaES"])},
                           esToEs={"geometry": set(["topology"])})


class TestParseTracerLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handle, cls.filename = tempfile.mkstemp(suffix=".log")
        with os.fdopen(handle, "w") as log:
            log.write(TRACER_LOG)
        cls.graph = parseTracerLog(cls.filename)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.filename)

    def testModuleDependencies(self):
        self.assertEqual(self.graph.edToEd["digis"], set(["rawData"]))
        self.assertEqual(self.graph.edToEd["soaClusters"], set(["digis"]))

    def testPathStatusIsAModuleDependency(self):
        self.assertEqual(self.graph.edToEd["l1tAlgoBlock"], set(["pSeed"]))

    def testEventSetupDependencies(self):
        # an EventSetup module without a label is named by its type
        self.assertEqual(self.graph.edToEs["soaClusters"],
                         set(["geometry", "UnlabelledESProducer"]))
        self.assertEqual(self.graph.esToEs["geometry"], set(["topology"]))

    def testTrailingSectionIsIgnored(self):
        # the last section lists consumed products, not modules, and must not
        # add any dependency
        self.assertEqual(self.graph.edToEd["soaClusters"], set(["digis"]))

    def testEmptyLogIsRejected(self):
        handle, empty = tempfile.mkstemp(suffix=".log")
        os.close(handle)
        try:
            self.assertRaises(RuntimeError, parseTracerLog, empty)
        finally:
            os.remove(empty)


class TestEventSetupClosure(unittest.TestCase):
    def testTransitiveReachability(self):
        needed = FRAMEWORK.eventSetupClosure(["soaClusters"])
        self.assertEqual(needed, set(["geometry", "topology"]))

    def testUnreachableIsNotIncluded(self):
        self.assertNotIn("unused", FRAMEWORK.eventSetupClosure(["soaClusters"]))


class TestCollectInputTags(unittest.TestCase):
    def testDescendsIntoNestedParameters(self):
        module = cms.EDProducer(
            "Test",
            plain=cms.InputTag("a"),
            nested=cms.PSet(inner=cms.InputTag("b")),
            vector=cms.VPSet(cms.PSet(inner=cms.InputTag("c"))),
            tags=cms.VInputTag(cms.InputTag("d"), "e"),
            untracked=cms.untracked.InputTag("f"),
            notATag=cms.string("g"))
        tags = []
        collectInputTags(module, "test", tags)
        self.assertEqual(sorted(tag.getModuleLabel() for tag, _ in tags),
                         ["a", "b", "c", "d", "e", "f"])


class TestModuleGraph(unittest.TestCase):
    def setUp(self):
        self.process = buildProcess()
        self.graph = ModuleGraph(self.process, FRAMEWORK)

    def testScheduledLabels(self):
        scheduled = scheduledLabels(self.process)
        self.assertIn("digis", scheduled)
        self.assertNotIn("rawData", scheduled)

    def testSeedsAreTheScheduledAlpakaModules(self):
        self.assertEqual(sorted(self.graph.seeds()), ["soaClusters", "soaHits"])

    def testUnscheduledAlpakaModuleIsNotASeed(self):
        self.assertIn("offClusters", self.graph.alpakaModules())
        self.assertNotIn("offClusters", self.graph.seeds())

    def testClassify(self):
        self.assertEqual(self.graph.classify("digis"), "module")
        self.assertEqual(self.graph.classify("rawData"), "defined-not-scheduled")
        self.assertEqual(self.graph.classify("pSeed"), "path")
        self.assertEqual(self.graph.classify("geometry"), "eventsetup")
        self.assertEqual(self.graph.classify("nowhere"), "not-in-process")

    def testClosureStopsAtAnUnscheduledModule(self):
        provenance, boundary, _ = self.graph.closure(self.graph.seeds())
        self.assertIn("digis", provenance)
        self.assertNotIn("rawData", provenance)
        self.assertEqual(boundary["rawData"]["kind"], "defined-not-scheduled")

    def testClosureIgnoresDownstreamAndUnrelatedModules(self):
        provenance, _, _ = self.graph.closure(self.graph.seeds())
        self.assertNotIn("legacyHits", provenance)
        self.assertNotIn("unrelated", provenance)

    def testClosureKeepsAPathWhoseStatusIsConsumed(self):
        provenance, _, paths = self.graph.closure(["hltL1GTAcceptFilter"])
        self.assertIn("l1tAlgoBlock", provenance)
        self.assertIn("pSeed", paths)
        # the modules of a Path that is kept join the closure
        self.assertIn("l1Seed", provenance)

    def testTopologicalOrder(self):
        order = self.graph.topologicalOrder(["soaHits", "soaClusters", "digis"])
        self.assertLess(order.index("digis"), order.index("soaClusters"))
        self.assertLess(order.index("soaClusters"), order.index("soaHits"))


class TestReduceToHeterogeneous(unittest.TestCase):
    def setUp(self):
        self.process = buildProcess()
        self.manifest = reduceToHeterogeneous(self.process, FRAMEWORK)

    def testPathIsTheHltStructureAroundTheSeeds(self):
        self.assertEqual(self.manifest["pathOrder"],
                         ["HLTBeginSequence", "hltL1GTAcceptFilter",
                          "soaClusters", "soaHits", "HLTEndSequence"])

    def testScheduleKeepsTheRequiredPathAndTheBookkeepingPaths(self):
        self.assertEqual(self.manifest["schedule"],
                         ["pSeed", "DST_HeterogeneousReco",
                          "HLTriggerFinalPath", "HLTAnalyzerEndpath"])

    def testDependenciesAreOnTheTask(self):
        task = set(self.process.DST_HeterogeneousRecoTask.moduleNames())
        self.assertIn("digis", task)
        self.assertIn("l1tAlgoBlock", task)
        # a module on a Path that is kept whole stays there, not on the Task
        self.assertNotIn("l1Seed", task)
        # the seeds are scheduled on the Path, not on the Task
        self.assertNotIn("soaClusters", task)

    def testModulesOutsideTheClosureAreDropped(self):
        for label in ("legacyHits", "unrelated", "offClusters", "rawData"):
            self.assertFalse(hasattr(self.process, label),
                             "%s should have been pruned" % label)

    def testPathsOutsideTheClosureAreDropped(self):
        self.assertFalse(hasattr(self.process, "reco"))
        self.assertFalse(hasattr(self.process, "other"))
        self.assertTrue(hasattr(self.process, "pSeed"))

    def testUnreachableEventSetupModulesAreDropped(self):
        kept = self.manifest["keptEventSetup"]
        self.assertIn("geometry", kept)
        self.assertIn("topology", kept)
        # kept whatever the graph says, the backend is chosen at run time
        self.assertIn("alpakaES", kept)
        self.assertIn("unused", self.manifest["droppedEventSetup"])
        self.assertIn("emptySource", self.manifest["droppedEventSetup"])

    def testManifestReportsUnscheduledAlpakaModules(self):
        self.assertEqual(self.manifest["unscheduledAlpakaModules"], ["offClusters"])

    def testSeedsAreReported(self):
        self.assertEqual(sorted(seed["label"] for seed in self.manifest["seeds"]),
                         ["soaClusters", "soaHits"])


class TestWithoutSeeds(unittest.TestCase):
    def testAConfigurationWithoutAlpakaModulesIsRejected(self):
        process = cms.Process("TEST")
        process.source = cms.Source("EmptySource")
        process.plain = cms.EDProducer("Plain")
        process.p = cms.Path(process.plain)
        process.schedule = cms.Schedule(process.p)
        self.assertRaises(RuntimeError, reduceToHeterogeneous, process,
                          FrameworkGraph())


if __name__ == "__main__":
    unittest.main(verbosity=2)
