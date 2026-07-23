"""RECO-side customise for the pileup truth chain: build the merged (signal+pileup)
logical graph from the MixingModule accumulator's raw TruthGraph (label mix) and
resolve its SimHit associations against the mixed rechits. Pairs with
mixedTruthGraphCustomize.addTruthGraphAccumulator at DIGI, giving a pileup-aware
truth graph at RECO."""

import FWCore.ParameterSet.Config as cms


def customise(process):
    from PhysicsTools.TruthInfo.truthGraphValidation_cff import (
        truthLogicalGraphProducer,
        detIdToRecHitMapProducer,
        truthLogicalGraphHitIndexProducer,
    )
    # The merged raw TruthGraph comes from the mixing accumulator, not the
    # standalone truthGraphProducer.
    process.truthLogicalGraphProducer = truthLogicalGraphProducer.clone(
        src=cms.InputTag("mix"),
        # The hitless-subgraph pruning decides which SimTracks left a calo hit; feed it
        # the merged (signal+pileup) HGCal sim-hits, else every pileup subgraph looks
        # hitless (its hits are in mix:mergedHGCHits, not the signal-only g4SimHits) and
        # is pruned, leaving pileup branches empty. Matches the hit index below.
        simHitCollections=cms.VInputTag(cms.InputTag('mix', 'mergedHGCHits')),
    )
    process.detIdToRecHitMapProducer = detIdToRecHitMapProducer
    # The hit index also reads the RAW merged graph (for trackId->node); point it at mix.
    process.truthLogicalGraphHitIndexProducer = truthLogicalGraphHitIndexProducer.clone(
        rawSrc=cms.InputTag('mix'),
        # Read the merged (signal+pileup) HGCal sim-hits from the accumulator, each
        # tagged with its sub-event EncodedEventId, instead of the signal-only
        # g4SimHits (which lack pileup at RECO).
        simHitCollections=cms.VInputTag(cms.InputTag('mix', 'mergedHGCHits')),
    )

    process.truthMixedRecoPath = cms.Path(
        process.truthLogicalGraphProducer
        + process.detIdToRecHitMapProducer
        + process.truthLogicalGraphHitIndexProducer
    )
    process.schedule.append(process.truthMixedRecoPath)

    for out in process.outputModules_().values():
        out.outputCommands.extend([
            "keep *_truthLogicalGraphProducer_*_*",
        ])
        # The hit index is an intermediate graph footprint, regenerable from the graph
        # plus mix:mergedHGCHits; persisting it costs ~9.7 MB/event (subgraph-hit CSR
        # over every retained pileup particle), so it is kept out of the event content.
    return process
