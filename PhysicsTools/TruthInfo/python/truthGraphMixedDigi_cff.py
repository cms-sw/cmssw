# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# DIGI-step pileup-aware truth, wired under the enableTruth modifier: the
# TruthGraphAccumulator (registered in the MixingModule digitizers) builds the
# merged signal+pileup raw TruthGraph during mixing, then the logical graph and an
# UNRESOLVED per-particle per-cell hit index are built right after mixing, where the
# merged simHits are live. recHitMap="" leaves the index unresolved because the
# shared-energy association matches by DetId, so the merged simHits never have to
# cross the DIGI->RECO boundary.
#
# DEFAULT SCOPE: full detector - calo (HGCal + ECAL + HCAL) + MTD + muon + tracker.
# Track-based candidate matching (a TICLCandidate is two-channel: calo shared energy
# plus tracker shared hits) needs the tracker channel, so it is in the default. The
# tracker is the largest sim-hit family and the dominant cost, so the reduced variant
# (mixedTruthGraphCustomize.customiseTruthReduced) drops it for cost-sensitive runs,
# leaving calo + MTD + muon. MTD sim-hits are captured here; the MTD channel of the
# hit index is resolved at RECO (it needs reco MTD cluster associations).

import FWCore.ParameterSet.Config as cms


def _tags(*names):
    return cms.VInputTag(*[cms.InputTag("g4SimHits", n) for n in names])


# Accumulator PSet added to the mixing digitizers under enableTruth (digitizers_cfi).
truthGraphAccumulator = cms.PSet(
    accumulatorType=cms.string("TruthGraphAccumulator"),
    simTracks=cms.InputTag("g4SimHits"),
    simVertices=cms.InputTag("g4SimHits"),
    genEventHepMC3=cms.InputTag("generatorSmeared"),
    genEventHepMC=cms.InputTag("generatorSmeared"),
    caloHits=_tags("HGCHitsEE", "HGCHitsHEfront", "HGCHitsHEback"),
    ecalHits=_tags("EcalHitsEB"),
    hcalHits=_tags("HcalHits"),
    trackerHits=_tags(
        "TrackerHitsPixelBarrelLowTof", "TrackerHitsPixelBarrelHighTof",
        "TrackerHitsPixelEndcapLowTof", "TrackerHitsPixelEndcapHighTof",
        "TrackerHitsTIBLowTof", "TrackerHitsTIBHighTof",
        "TrackerHitsTIDLowTof", "TrackerHitsTIDHighTof",
        "TrackerHitsTOBLowTof", "TrackerHitsTOBHighTof",
        "TrackerHitsTECLowTof", "TrackerHitsTECHighTof",
    ),   # full detector by default; customiseTruthReduced empties this
    muonHits=_tags("MuonDTHits", "MuonCSCHits", "MuonRPCHits", "MuonGEMHits", "MuonME0Hits"),
    mtdHits=_tags("FastTimerHitsBarrel", "FastTimerHitsEndcap"),
    pileupBunchCrossings=cms.vint32(0),   # in-time pileup for the per-particle graph
    collapsePileupGen=cms.bool(True),    # pileup keeps stable GEN particles only
    collapseSignalGen=cms.bool(False),   # signal keeps the full HepMC decay chain, which
                                         # selection presets seed on
    collapseGenShower=cms.bool(True),    # contract the parton shower and the intermediate
                                         # resonance copies out of that chain, keeping ancestry

    computeCellEnergyBudget=cms.bool(False),  # prototype energy-budget map, off by default
)

# The post-mixing build: logical graph + unresolved hit index from the mixed raw
# graph (label mix) and the accumulator's merged simHits.
from Validation.Configuration.truthPrevalidation_cff import (
    truthLogicalGraphProducer as _truthLogicalGraphProducer,
    truthLogicalGraphHitIndexProducer as _truthLogicalGraphHitIndexProducer,
)

# The hitless-subgraph pruning must see every detector the hit index reads, and read
# the merged (signal+pileup) products: a calorimeter or a sub-event left out here has
# its particles pruned as hitless even though the index would have given them hits.
truthLogicalGraphProducer = _truthLogicalGraphProducer.clone(
    src=cms.InputTag("mix"),
    simHitCollections=cms.VInputTag(
        cms.InputTag("mix", "mergedHGCHits"),
        cms.InputTag("mix", "mergedEcalHits"),
        cms.InputTag("mix", "mergedHcalHits"),
    ),
    trackerSimHitCollections=cms.VInputTag(cms.InputTag("mix", "mergedTrackerHits")),
    muonSimHitCollections=cms.VInputTag(cms.InputTag("mix", "mergedMuonHits")),
)

truthLogicalGraphHitIndexProducer = _truthLogicalGraphHitIndexProducer.clone(
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("mix"),
    recHitMap=cms.InputTag(""),   # UNRESOLVED: association is by DetId
    # HCAL simulation DetIds are in packed test numbering; HcalHitRelabeller converts
    # them to the reco HcalDetIds the association matches on. ECAL barrel and the Run4
    # HGCAL geometries carry reco DetIds already, so only the HCAL switch is set here.
    doHcalRelabelling=cms.bool(True),
    subdetectors=cms.vstring("Calo", "Muon", "Tracker"),  # full; MTD resolved at RECO
    simHitCollections=cms.VInputTag(
        cms.InputTag("mix", "mergedHGCHits"),
        cms.InputTag("mix", "mergedEcalHits"),
        cms.InputTag("mix", "mergedHcalHits"),
    ),
    trackerSimHitCollections=cms.VInputTag(cms.InputTag("mix", "mergedTrackerHits")),  # customiseTruthReduced empties this
    muonSimHitCollections=cms.VInputTag(cms.InputTag("mix", "mergedMuonHits")),
)

# Task so the producers run at DIGI on demand (triggered by the output keeps); added
# to pdigiTask under enableTruth in Configuration/StandardSequences/Digi_cff.
truthGraphMixedDigiTask = cms.Task(
    truthLogicalGraphProducer,
    truthLogicalGraphHitIndexProducer,
)
