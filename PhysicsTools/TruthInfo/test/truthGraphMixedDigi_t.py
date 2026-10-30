#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""The collapsed pileup record keeps exactly the species the reconstructable levels
stop at. The hitless-subgraph pruning reads every sim-hit family the accumulator merges."""

import unittest

import FWCore.ParameterSet.Config as cms
import PhysicsTools.TruthInfo.truthGraphMixedDigi_cff as digi
from PhysicsTools.TruthInfo.mixedTruthGraphCustomize import customiseTruthReduced

# The accumulator input of each sim-hit family, the merged product it writes, and the
# pruning parameter that reads that product.
_hitFamilies = (
    ("caloHits", "mergedHGCHits", "simHitCollections"),
    ("ecalHits", "mergedEcalHits", "simHitCollections"),
    ("hcalHits", "mergedHcalHits", "simHitCollections"),
    ("trackerHits", "mergedTrackerHits", "trackerSimHitCollections"),
    ("muonHits", "mergedMuonHits", "muonSimHitCollections"),
    ("mtdHits", "mergedMtdHits", "mtdSimHitCollections"),
)


def reconstructable(producer):
    return list(producer.postProcessing.reconstructablePdgIds)


class TestKeptSpeciesMatchTheLevels(unittest.TestCase):
    def test_default_configuration(self):
        kept = list(digi.truthGraphAccumulator.collapsedGenKeptPdgIds)
        self.assertEqual(kept, digi.reconstructablePdgIds)
        self.assertEqual(reconstructable(digi.truthLogicalGraphProducer), kept)


class TestPruningReadsEveryMergedHitFamily(unittest.TestCase):
    def check(self, accumulator, producer):
        for accumulatorParameter, instance, pruningParameter in _hitFamilies:
            read = [(tag.getModuleLabel(), tag.getProductInstanceLabel())
                    for tag in getattr(producer, pruningParameter, [])]
            merged = len(getattr(accumulator, accumulatorParameter)) > 0
            self.assertEqual(("mix", instance) in read, merged, f"{instance} in {pruningParameter}")

    def test_default_configuration(self):
        self.check(digi.truthGraphAccumulator, digi.truthLogicalGraphProducer)

    def test_reduced_configuration(self):
        process = cms.Process("TEST")
        process.mix = cms.EDProducer("MixingModule", digitizers=cms.PSet(truthGraph=digi.truthGraphAccumulator.clone()))
        process.truthLogicalGraphProducer = digi.truthLogicalGraphProducer.clone()
        process.truthLogicalGraphHitIndexProducer = digi.truthLogicalGraphHitIndexProducer.clone()
        customiseTruthReduced(process)
        self.check(process.mix.digitizers.truthGraph, process.truthLogicalGraphProducer)


if __name__ == "__main__":
    unittest.main()
