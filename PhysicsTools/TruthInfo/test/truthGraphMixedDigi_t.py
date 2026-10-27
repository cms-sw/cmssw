#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""The collapsed pileup record keeps exactly the species the reconstructable levels
stop at."""

import unittest

import PhysicsTools.TruthInfo.truthGraphMixedDigi_cff as digi


def reconstructable(producer):
    return list(producer.postProcessing.reconstructablePdgIds)


class TestKeptSpeciesMatchTheLevels(unittest.TestCase):
    def test_default_configuration(self):
        kept = list(digi.truthGraphAccumulator.collapsedGenKeptPdgIds)
        self.assertEqual(kept, digi.reconstructablePdgIds)
        self.assertEqual(reconstructable(digi.truthLogicalGraphProducer), kept)


if __name__ == "__main__":
    unittest.main()
