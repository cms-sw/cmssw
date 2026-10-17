# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Customise of the pileup-aware truth that the DIGI step builds by default
# (truthGraphMixedDigi_cff, registered under enableTruth).

import FWCore.ParameterSet.Config as cms


def customiseTruthReduced(process):
    """Reduced variant: drop the Tracker channel from the DIGI-built truth, leaving
    calo (HGCal + ECAL + HCAL) + MTD + muon. Apply it at the DIGI step for a
    cost-sensitive production that does not need track-based candidate matching (the
    tracker is the largest sim-hit family and dominates the DIGI cost)."""
    acc = process.mix.digitizers.truthGraph
    acc.trackerHits = cms.VInputTag()
    idx = process.truthLogicalGraphHitIndexProducer
    idx.subdetectors = cms.vstring("Calo", "Muon")
    # The pruning's detector scope stays equal to the index's, otherwise it prunes on
    # tracker hits that are no longer accumulated.
    process.truthLogicalGraphProducer.trackerSimHitCollections = cms.VInputTag()
    return process
