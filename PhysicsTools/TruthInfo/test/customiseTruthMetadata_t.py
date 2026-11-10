#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
"""customiseTruthMetadata picks the graph labels of the job, and the truth keeps go only to
the EDM output modules that write the GEN events."""

import unittest

import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.customiseTruthMetadata import (addTruthMetadata, customiseTruthMetadata,
                                                           truthOutputModules)


def _output(cppType, tier):
    return cms.OutputModule(cppType, outputCommands=cms.untracked.vstring("drop *"),
                            dataset=cms.untracked.PSet(dataTier=cms.untracked.string(tier)))


class TestCustomiseTruthMetadata(unittest.TestCase):
    def testOutputModules(self):
        process = cms.Process("TEST")
        process.GENoutput = _output("PoolOutputModule", "GEN")
        process.LHEoutput = _output("PoolOutputModule", "LHE")
        process.NANOAODGENoutput = _output("NanoAODOutputModule", "NANOAODSIM")
        self.assertEqual([out.label_() for out in truthOutputModules(process)], ["GENoutput"])

    def testNeedsOneGraph(self):
        process = cms.Process("TEST")
        self.assertRaises(RuntimeError, customiseTruthMetadata, process)
        process.truthLogicalGraphProducer = cms.EDProducer("TruthLogicalGraphProducer")
        process.truthGenOnlyLogicalGraphProducer = cms.EDProducer("TruthLogicalGraphProducer")
        self.assertRaises(RuntimeError, customiseTruthMetadata, process)

    def testDigiLabels(self):
        process = cms.Process("TEST")
        process.truthLogicalGraphProducer = cms.EDProducer("TruthLogicalGraphProducer")
        process.out = _output("PoolOutputModule", "GEN-SIM-DIGI-RAW")
        customiseTruthMetadata(process)
        self.assertEqual(process.truthGraphMetadata.rawGraphProducer.value(), "mix")
        self.assertIn("keep *_truthGraphMetadata_*_*", process.out.outputCommands)
        self.assertRaises(RuntimeError, addTruthMetadata, process, "a", "b")


if __name__ == "__main__":
    unittest.main()
