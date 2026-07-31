# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

# Phase-A pileup customise: enable the SimTrack/SimVertex crossing frames and run
# TruthGraphMixedProducer in the DIGI step (the only place the transient
# CrossingFrame<SimTrack/SimVertex> products live), then keep the compact mixed
# raw TruthGraph in the output so downstream steps can read signal+pileup truth.

import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn


def addMixedTruthGraph(process):
    # makeCrossingFrame=True for SimTrack/SimVertex (transient, in-process only).
    process = setCrossingFrameOn(process)

    process.truthGraphMixedProducer = cms.EDProducer(
        "TruthGraphMixedProducer",
        simTracks=cms.InputTag("mix", "g4SimHits"),
        simVertices=cms.InputTag("mix", "g4SimHits"),
    )

    process.truthGraphMixedPath = cms.Path(process.truthGraphMixedProducer)
    if process.schedule is not None:
        process.schedule.append(process.truthGraphMixedPath)

    for out in process.outputModules_().values():
        out.outputCommands.append("keep *_truthGraphMixedProducer_*_*")

    return process


def addTruthGraphAccumulator(process,
                             pileupBunchCrossings=(0,),
                             collapsePileupGen=True,
                             includeTrackingHits=True,
                             computeCellEnergyBudget=False):
    """Phase-B (B1): register TruthGraphAccumulator inside the MixingModule.

    The accumulator builds the mixed (signal + pileup) raw TruthGraph from the
    native per-sub-event SimTrack/SimVertex collections. By default only in-time
    pileup (bx 0) is included; pass pileupBunchCrossings to widen. The mixed graph
    is kept in the output as TruthGraph_mix__<process>.

    By default (includeTrackingHits=True) the accumulator captures the full detector
    (calo + tracker + muon + MTD). Pass includeTrackingHits=False for a calo-only
    graph (HGCAL plus barrel ECAL/HCAL): the tracker is the largest sim-hit family at
    PU200 and dominates the event size (the merged tracker PSimHits alone are tens of
    MB/event), so dropping it is the main cost lever.
    """
    def tags(*names):
        return cms.VInputTag(*[cms.InputTag("g4SimHits", n) for n in names])

    # Tracking detectors only when explicitly requested; empty (calo-only) otherwise.
    trackerHits = cms.VInputTag()
    muonHits = cms.VInputTag()
    mtdHits = cms.VInputTag()
    if includeTrackingHits:
        trackerHits = tags(
            "TrackerHitsPixelBarrelLowTof", "TrackerHitsPixelBarrelHighTof",
            "TrackerHitsPixelEndcapLowTof", "TrackerHitsPixelEndcapHighTof",
            "TrackerHitsTIBLowTof", "TrackerHitsTIBHighTof",
            "TrackerHitsTIDLowTof", "TrackerHitsTIDHighTof",
            "TrackerHitsTOBLowTof", "TrackerHitsTOBHighTof",
            "TrackerHitsTECLowTof", "TrackerHitsTECHighTof",
        )
        muonHits = tags("MuonDTHits", "MuonCSCHits", "MuonRPCHits", "MuonGEMHits", "MuonME0Hits")
        mtdHits = tags("FastTimerHitsBarrel", "FastTimerHitsEndcap")

    process.mix.digitizers.truthGraph = cms.PSet(
        accumulatorType=cms.string("TruthGraphAccumulator"),
        simTracks=cms.InputTag("g4SimHits"),
        simVertices=cms.InputTag("g4SimHits"),
        genEventHepMC3=cms.InputTag("generatorSmeared"),
        genEventHepMC=cms.InputTag("generatorSmeared"),
        caloHits=tags("HGCHitsEE", "HGCHitsHEfront", "HGCHitsHEback"),
        # Barrel calorimeters, kept in separate products so the RECO consumer applies
        # the right sim-to-reco DetId relabelling per collection (ECAL barrel needs
        # none, HCAL uses HcalHitRelabeller).
        ecalHits=tags("EcalHitsEB"),
        hcalHits=tags("HcalHits"),
        trackerHits=trackerHits,
        muonHits=muonHits,
        mtdHits=mtdHits,
        pileupBunchCrossings=cms.vint32(*pileupBunchCrossings),
        collapsePileupGen=cms.bool(collapsePileupGen),
        collapseSignalGen=cms.bool(False),
        collapseGenShower=cms.bool(True),
        # Prototype energy-budget closure: sum per-cell HGCal energy over ALL bunch
        # crossings (in-time + out-of-time) into cellTotalEnergy/cellTotalDetId, so
        # "untracked" energy (out-of-time pileup + dropped in-time) can be measured
        # as total minus the in-time truth-branch energy.
        computeCellEnergyBudget=cms.bool(computeCellEnergyBudget),
    )

    # Persistence is owned by truthEventContent_cff (the compact/full verbosity
    # levels), applied via customiseTruthDigi. The accumulator products
    # (TruthGraph_mix, mix:merged*Hits) are kept only at the 'full' level; the
    # compact default persists the graph + unresolved index built below instead.
    return process


def buildCompactTruthAtDigi(process, includeTrackingHits=True):
    """Build the stage-independent truth right after mixing, where the merged
    signal+pileup simHits are live: the logical graph plus an UNRESOLVED
    per-particle per-cell hit index.

    The index is built with recHitMap="" (recHitIndex left invalid) because the
    shared-energy association (BranchHitAssociator) matches by DetId, not by
    recHitIndex. The same unresolved index therefore serves any later stage (L1,
    HLT, offline RECO) that exposes its reco objects as (DetId, fraction), and the
    bulky merged simHits never have to cross the DIGI->RECO boundary in the compact
    content.

    Channels: Calo (plus Tracker and Muon under includeTrackingHits) are pure
    simHit reads and are built here. MTD is intentionally left out: its index needs
    the reco Mtd cluster associations, which are RECO-stage products, so it cannot be
    filled at DIGI (muon/MTD recHit linking is follow-up work in any case).
    """
    from Validation.Configuration.truthPrevalidation_cff import (
        truthLogicalGraphProducer,
        truthLogicalGraphHitIndexProducer,
    )

    # Calorimeter simHits for the Calo channel: the merged (signal+pileup)
    # products the accumulator wrote, one per calo family for the per-collection
    # DetId relabelling (HGCAL unpack, HCAL HcalHitRelabeller, ECAL none).
    caloSimHits = cms.VInputTag(
        cms.InputTag("mix", "mergedHGCHits"),
        cms.InputTag("mix", "mergedEcalHits"),
        cms.InputTag("mix", "mergedHcalHits"),
    )
    subdetectors = ["Calo"]
    trackerSimHits = cms.VInputTag()
    muonSimHits = cms.VInputTag()
    if includeTrackingHits:
        subdetectors = ["Calo", "Tracker", "Muon"]
        trackerSimHits = cms.VInputTag(cms.InputTag("mix", "mergedTrackerHits"))
        muonSimHits = cms.VInputTag(cms.InputTag("mix", "mergedMuonHits"))

    # Logical graph from the mixed raw graph. The hitless-subgraph pruning reads the
    # same merged collections as the index below: a calorimeter or a sub-event left
    # out here has its particles pruned as hitless even though they do carry hits.
    process.truthLogicalGraphProducer = truthLogicalGraphProducer.clone(
        src=cms.InputTag("mix"),
        simHitCollections=caloSimHits,
        trackerSimHitCollections=trackerSimHits,
    )

    process.truthLogicalGraphHitIndexProducer = truthLogicalGraphHitIndexProducer.clone(
        src=cms.InputTag("truthLogicalGraphProducer"),
        rawSrc=cms.InputTag("mix"),
        recHitMap=cms.InputTag(""),   # UNRESOLVED: association is by DetId
        subdetectors=cms.vstring(*subdetectors),
        simHitCollections=caloSimHits,
        trackerSimHitCollections=trackerSimHits,
        muonSimHitCollections=muonSimHits,
    )

    process.truthCompactBuildPath = cms.Path(
        process.truthLogicalGraphProducer
        + process.truthLogicalGraphHitIndexProducer
    )
    if process.schedule is not None:
        process.schedule.append(process.truthCompactBuildPath)

    return process


def customiseTruthDigi(process,
                       level='compact',
                       pileupBunchCrossings=(0,),
                       collapsePileupGen=True,
                       includeTrackingHits=True):
    """DIGI-step entry point for the pileup-aware truth graph: register the
    accumulator, build the compact stage-independent truth (graph + unresolved hit
    index) where the merged simHits are live, and apply the event-content level.

    level='compact' (default) persists the logical graph, the unresolved index and
    the raw graph; level='full' also keeps the merged simHits (for off-geometry
    re-association, e.g. L1)."""
    process = addTruthGraphAccumulator(process,
                                       pileupBunchCrossings=pileupBunchCrossings,
                                       collapsePileupGen=collapsePileupGen,
                                       includeTrackingHits=includeTrackingHits)
    process = buildCompactTruthAtDigi(process, includeTrackingHits=includeTrackingHits)

    from PhysicsTools.TruthInfo.truthEventContent_cff import setTruthEventContent
    process = setTruthEventContent(process, level=level, includeTrackingHits=includeTrackingHits)
    return process


def customiseTruthReduced(process):
    """Reduced variant: drop the Tracker channel from the DIGI-built truth, leaving
    calo (HGCal + ECAL + HCAL) + MTD + muon. The default scope
    (truthGraphMixedDigi_cff) is the full detector; apply this at the DIGI step for a
    cost-sensitive production that does not need track-based candidate matching (the
    tracker is the largest sim-hit family and dominates the DIGI cost)."""
    acc = process.mix.digitizers.truthGraph
    acc.trackerHits = cms.VInputTag()
    idx = process.truthLogicalGraphHitIndexProducer
    idx.subdetectors = cms.vstring("Calo", "Muon")
    idx.trackerSimHitCollections = cms.VInputTag()
    # The pruning's detector scope stays equal to the index's, otherwise it prunes on
    # tracker hits that are no longer accumulated.
    process.truthLogicalGraphProducer.trackerSimHitCollections = cms.VInputTag()
    return process


def customiseTruthDigiEnergyBudget(process):
    """Prototype entry point: customiseTruthDigi plus the per-cell all-bunch-crossing
    HGCal energy budget (cellTotalEnergy/cellTotalDetId products), for measuring the
    out-of-time / untracked calorimeter energy fraction at pileup. The accumulator
    logs allBx/inTime/untracked per event."""
    process = customiseTruthDigi(process)
    process.mix.digitizers.truthGraph.computeCellEnergyBudget = cms.bool(True)
    for out in process.outputModules_().values():
        out.outputCommands.extend([
            'keep *_mix_cellTotalDetId_*',
            'keep *_mix_cellTotalEnergy_*',
            'keep *_mix_cellInTimeEnergy_*',
        ])
    return process
