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
                             includeTrackingHits=False):
    """Phase-B (B1): register TruthGraphAccumulator inside the MixingModule.

    The accumulator builds the mixed (signal + pileup) raw TruthGraph from the
    native per-sub-event SimTrack/SimVertex collections. By default only in-time
    pileup (bx 0) is included; pass pileupBunchCrossings to widen. The mixed graph
    is kept in the output as TruthGraph_mix__<process>.

    By default the accumulator captures only the CALORIMETER sim-hits (HGCAL plus
    barrel ECAL/HCAL). The tracking detectors (tracker, muon chambers, MTD) are the
    largest sim-hit family at PU200 and dominate the event size (the merged tracker
    PSimHits alone are tens of MB/event), so they are OFF by default. Pass
    includeTrackingHits=True for a full-detector truth graph.
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
    )

    for out in process.outputModules_().values():
        out.outputCommands.append("keep TruthGraph_mix_*_*")
        # The merged sim-hit collections are the union of signal + all kept pileup hits,
        # and must bridge a split DIGI->RECO job (the pileup hits are gone after mixing;
        # the RECO customise does not re-keep them). Drop these lines for a single-job
        # DIGI+RECO where the hit index is built in the same process.
        out.outputCommands.append("keep *_mix_mergedHGCHits_*")
        out.outputCommands.append("keep *_mix_mergedEcalHits_*")
        out.outputCommands.append("keep *_mix_mergedHcalHits_*")
        if includeTrackingHits:
            out.outputCommands.append("keep *_mix_mergedTrackerHits_*")
            out.outputCommands.append("keep *_mix_mergedMuonHits_*")
            out.outputCommands.append("keep *_mix_mergedMtdHits_*")

    return process
