#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""The collapsed pileup record keeps exactly the species the reconstructable levels
stop at. The hitless-subgraph pruning reads every sim-hit family the accumulator merges,
and the hit index fills the channel of each one."""

import unittest

import FWCore.ParameterSet.Config as cms
import PhysicsTools.TruthInfo.truthGraphMixedDigi_cff as digi
from PhysicsTools.TruthInfo.mixedTruthGraphCustomize import customiseTruthReduced

# The accumulator input of each sim-hit family, the merged product it writes, the
# pruning parameter that reads that product, and the hit index channel of the family.
_hitFamilies = (
    ("caloHits", "mergedHGCHits", "simHitCollections", "Calo"),
    ("ecalHits", "mergedEcalHits", "simHitCollections", "Calo"),
    ("hcalHits", "mergedHcalHits", "simHitCollections", "Calo"),
    ("trackerHits", "mergedTrackerHits", "trackerSimHitCollections", "Tracker"),
    ("muonHits", "mergedMuonHits", "muonSimHitCollections", "Muon"),
    ("mtdHits", "mergedMtdHits", "mtdSimHitCollections", "MTD"),
)


def reconstructable(producer):
    return list(producer.postProcessing.reconstructablePdgIds)


class TestKeptSpeciesMatchTheLevels(unittest.TestCase):
    def test_default_configuration(self):
        kept = list(digi.truthGraphAccumulator.collapsedGenKeptPdgIds)
        self.assertEqual(kept, digi.reconstructablePdgIds)
        self.assertEqual(reconstructable(digi.truthLogicalGraphProducer), kept)


class TestPruningReadsEveryMergedHitFamily(unittest.TestCase):
    def check(self, accumulator, producer, index):
        for accumulatorParameter, instance, pruningParameter, channel in _hitFamilies:
            read = [(tag.getModuleLabel(), tag.getProductInstanceLabel())
                    for tag in getattr(producer, pruningParameter, [])]
            merged = len(getattr(accumulator, accumulatorParameter)) > 0
            self.assertEqual(("mix", instance) in read, merged, f"{instance} in {pruningParameter}")
            self.assertEqual(channel in index.subdetectors, merged, f"{channel} in subdetectors")

    def test_default_configuration(self):
        self.check(digi.truthGraphAccumulator, digi.truthLogicalGraphProducer, digi.truthLogicalGraphHitIndexProducer)

    def test_reduced_configuration(self):
        process = cms.Process("TEST")
        process.mix = cms.EDProducer("MixingModule", digitizers=cms.PSet(truthGraph=digi.truthGraphAccumulator.clone()))
        process.truthLogicalGraphProducer = digi.truthLogicalGraphProducer.clone()
        process.truthLogicalGraphHitIndexProducer = digi.truthLogicalGraphHitIndexProducer.clone()
        customiseTruthReduced(process)
        self.check(process.mix.digitizers.truthGraph, process.truthLogicalGraphProducer,
                   process.truthLogicalGraphHitIndexProducer)


if __name__ == "__main__":
    unittest.main()
