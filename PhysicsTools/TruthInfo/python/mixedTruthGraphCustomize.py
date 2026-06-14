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
                             collapsePileupGen=True):
    """Phase-B (B1): register TruthGraphAccumulator inside the MixingModule.

    The accumulator builds the mixed (signal + pileup) raw TruthGraph from the
    native per-sub-event SimTrack/SimVertex collections. By default only in-time
    pileup (bx 0) is included; pass pileupBunchCrossings to widen. The mixed graph
    is kept in the output as TruthGraph_mix__<process>.
    """
    process.mix.digitizers.truthGraph = cms.PSet(
        accumulatorType=cms.string("TruthGraphAccumulator"),
        simTracks=cms.InputTag("g4SimHits"),
        simVertices=cms.InputTag("g4SimHits"),
        genEventHepMC3=cms.InputTag("generatorSmeared"),
        genEventHepMC=cms.InputTag("generatorSmeared"),
        pileupBunchCrossings=cms.vint32(*pileupBunchCrossings),
        collapsePileupGen=cms.bool(collapsePileupGen),
        collapseSignalGen=cms.bool(False),
    )

    for out in process.outputModules_().values():
        out.outputCommands.append("keep TruthGraph_mix_*_*")

    return process
