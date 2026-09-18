#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
"""The preset customise sets the graph producer, the targets producer and every module that
reads the signal seeds, so a validator books its signal folders from the same preset."""

import unittest

import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.customiseTruthPreset import applyTruthPreset, customiseTruthPreset, resolvePreset
from Validation.Configuration.truthPrevalidation_cff import truthLogicalGraphProducer
from SimGeneral.TruthGraphAssociatorProducers.truthGraphAssociators_cff import truthBranchTargets


def _process():
    """A process carrying the real modules, so a renamed parameter fails here: the graph
    producer, the targets producer and one validator that books signal folders."""
    process = cms.Process("TEST")
    process.truthLogicalGraphProducer = truthLogicalGraphProducer.clone()
    process.truthBranchTargets = truthBranchTargets.clone()
    process.aTruthValidator = _validatorWithSeeds()
    return process


def _validatorWithSeeds():
    """The first truth validator of the DQM sequence that carries seed parameters."""
    from Validation.TruthInfo.truthBranchValidation_cff import truthBranchValidationSequence
    import FWCore.ParameterSet.Config as config

    found = []
    truthBranchValidationSequence.visit(config.ModuleNodeVisitor(found))
    for module in found:
        if hasattr(module, "signalSeedPdgIds") and hasattr(module, "signalSeedHadronFlavors"):
            return module.clone()
    raise AssertionError("no truth validator carries the seed parameters any more")


class TestCustomiseTruthPreset(unittest.TestCase):
    def testPresetByName(self):
        process = applyTruthPreset(_process(), preset="top")
        selection = process.truthLogicalGraphProducer.postProcessing
        self.assertEqual(list(selection.seedPdgIds), [6, -6])
        self.assertTrue(selection.keepProductionSiblings.value())
        self.assertEqual(list(process.truthBranchTargets.signalSeedPdgIds), [6, -6])

    def testEverySeededModuleIsSet(self):
        # A validator books its signal folders from its own seeds, so the preset has to
        # reach it too: with empty seeds there the signal plots are never booked.
        process = applyTruthPreset(_process(), preset="ggf")
        for label in ("truthBranchTargets", "aTruthValidator"):
            module = getattr(process, label)
            self.assertEqual(list(module.signalSeedPdgIds), [25], label)
            self.assertEqual(list(module.signalSeedHadronFlavors), [], label)

        flavoured = applyTruthPreset(_process(), preset="heavyflavor")
        for label in ("truthBranchTargets", "aTruthValidator"):
            module = getattr(flavoured, label)
            self.assertEqual(list(module.signalSeedPdgIds), [], label)
            self.assertEqual(list(module.signalSeedHadronFlavors), [5], label)

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
