#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""The collapsed pileup record keeps exactly the species the reconstructable levels
stop at, in every configuration that builds the mixed truth graph."""

import unittest

import FWCore.ParameterSet.Config as cms

import PhysicsTools.TruthInfo.mixedTruthGraphCustomize as customise
import PhysicsTools.TruthInfo.truthGraphMixedDigi_cff as digi


def reconstructable(producer):
    return list(producer.postProcessing.reconstructablePdgIds)


class TestKeptSpeciesMatchTheLevels(unittest.TestCase):
    def test_default_configuration(self):
        kept = list(digi.truthGraphAccumulator.collapsedGenKeptPdgIds)
        self.assertEqual(kept, digi.reconstructablePdgIds)
        self.assertEqual(reconstructable(digi.truthLogicalGraphProducer), kept)

    def test_customise(self):
        process = cms.Process("TEST")
        process.mix = cms.EDProducer("MixingModule", digitizers=cms.PSet())
        customise.addTruthGraphAccumulator(process)
        customise.buildCompactTruthAtDigi(process)
        kept = list(process.mix.digitizers.truthGraph.collapsedGenKeptPdgIds)
        self.assertEqual(kept, digi.reconstructablePdgIds)
        self.assertEqual(reconstructable(process.truthLogicalGraphProducer), kept)


if __name__ == "__main__":
    unittest.main()
