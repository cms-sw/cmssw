#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
"""The preset customise sets the graph producer and the targets producer together."""

import unittest

import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.customiseTruthPreset import applyTruthPreset, customiseTruthPreset, resolvePreset
from Validation.Configuration.truthPrevalidation_cff import truthLogicalGraphProducer
from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff import truthBranchTargets


def _process():
    """A process carrying the two real modules, so a renamed parameter fails here."""
    process = cms.Process("TEST")
    process.truthLogicalGraphProducer = truthLogicalGraphProducer.clone()
    process.truthBranchTargets = truthBranchTargets.clone()
    return process


class TestCustomiseTruthPreset(unittest.TestCase):
    def testPresetByName(self):
        process = applyTruthPreset(_process(), preset="top")
        selection = process.truthLogicalGraphProducer.postProcessing
        self.assertEqual(list(selection.seedPdgIds), [6, -6])
        self.assertTrue(selection.keepProductionSiblings.value())
        self.assertEqual(list(process.truthBranchTargets.signalSeedPdgIds), [6, -6])

    def testFragmentResolvesToPreset(self):
        self.assertEqual(resolvePreset(fragment="TTbar_14TeV_TuneCP5_cfi"), "top")
        process = applyTruthPreset(_process(), fragment="TTbar_14TeV_TuneCP5_cfi")
        self.assertEqual(list(process.truthLogicalGraphProducer.postProcessing.seedPdgIds), [6, -6])

    def testSeedsAgreeForEveryPreset(self):
        # The signal denominator is the preset's own signal object, so the two modules
        # carry the same species. [0] is the full-graph escape hatch, not a species.
        for preset in ("gun", "resonance", "vbf", "ggf", "vh", "top", "singletop", "diboson", "heavyflavor", "full"):
            process = applyTruthPreset(_process(), preset=preset)
            selection = process.truthLogicalGraphProducer.postProcessing
            expected = [p for p in list(selection.seedPdgIds) if p != 0]
            self.assertEqual(list(process.truthBranchTargets.signalSeedPdgIds), expected, preset)
            self.assertEqual(list(process.truthBranchTargets.signalSeedHadronFlavors),
                             list(selection.seedHadronFlavors), preset)

    def testFieldsOutsideThePresetSurvive(self):
        # A mixed job configures these; a preset must not take them back to the default.
        process = _process()
        process.truthLogicalGraphProducer.postProcessing.reconstructablePdgIds = cms.vint32(111, 310)
        process.truthLogicalGraphProducer.postProcessing.dropHitlessSimSubgraphs = cms.bool(False)
        applyTruthPreset(process, preset="ggf")
        selection = process.truthLogicalGraphProducer.postProcessing
        self.assertEqual(list(selection.reconstructablePdgIds), [111, 310])
        self.assertFalse(selection.dropHitlessSimSubgraphs.value())

    def testOverride(self):
        process = applyTruthPreset(_process(), preset="resonance", seedParentDepth=3, reconstructablePdgIds=[111, 130])
        selection = process.truthLogicalGraphProducer.postProcessing
        self.assertEqual(selection.seedParentDepth.value(), 3)
        self.assertEqual(list(selection.reconstructablePdgIds), [111, 130])

    def testDecayGroups(self):
        process = applyTruthPreset(_process(), fragment="ZMM_14TeV")
        groups = process.truthLogicalGraphProducer.postProcessing.decayPdgIdGroups
        self.assertEqual([list(g.pdgIds) for g in groups], [[13, -13]])

    def testRejectsWhatItCannotResolve(self):
        self.assertRaises(KeyError, resolvePreset, preset="nosuchpreset")
        self.assertRaises(ValueError, resolvePreset)
        self.assertRaises(ValueError, resolvePreset, preset="top", fragment="TTbar_14TeV")
        self.assertRaises(KeyError, applyTruthPreset, _process(), "top", None, nosuchfield=1)

    def testEnvironmentCustomise(self):
        import os

        for key in ("TRUTH_GRAPH_PRESET", "TRUTH_GRAPH_FRAGMENT"):
            os.environ.pop(key, None)
        process = customiseTruthPreset(_process())
        self.assertEqual(list(process.truthLogicalGraphProducer.postProcessing.seedPdgIds), [])

        os.environ["TRUTH_GRAPH_PRESET"] = "ggf"
        try:
            process = customiseTruthPreset(_process())
            self.assertEqual(list(process.truthLogicalGraphProducer.postProcessing.seedPdgIds), [25])
        finally:
            del os.environ["TRUTH_GRAPH_PRESET"]


if __name__ == "__main__":
    unittest.main()
