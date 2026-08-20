# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Event content for the MC-truth graph, with two verbosity levels.
#
# The stage-independent truth is built once at the mixing/DIGI step, where the
# merged signal+pileup simHits are live, as a logical graph plus a per-particle
# per-cell sim-energy hit index. The index is built UNRESOLVED (recHitMap="")
# because the shared-energy association (BranchHitAssociator) matches by DetId,
# not by recHitIndex: it keys the inverted index and the per-cell total-sim-energy
# denominator on DetId, so the same unresolved index serves any stage (L1, HLT,
# offline RECO) that later exposes its reco objects as (DetId, fraction). The
# recHitIndex field stays a convenience pointer, filled on demand only where a
# consumer wants index->recHit navigation.
#
#   compact (default): the logical graph, the unresolved hit index, and the raw
#     merged graph (TruthGraph_mix). The index is both the fraction numerator
#     (per-particle per-cell energy) and, summed over all particles per cell, the
#     denominator; correct hits-and-fractions shared energy at any stage whose
#     rechits share the cell-level DetId space. The raw graph is small (~3 MB/ev)
#     and is kept because the branch validators consume it (rawSrc, for the
#     trackId->particle map) and it lets the logical graph / index be rebuilt
#     offline.
#
#     INVARIANT: the persisted index must stay COMPLETE - every hit-leaving
#     contributor present, no pdgId pruning. The per-cell denominator is the sum
#     of the index hit energies over all particles; drop contributors and every
#     surviving fraction is biased high. Do not "optimize" the index by pruning it
#     to interesting species without also persisting a separate all-contributor
#     per-cell total map.
#
#   full: compact + the merged simHits (all contributors). The only level that
#     rebuilds the association at a DIFFERENT granularity (e.g. L1 trigger cells)
#     or with a different metric, because it keeps the raw deposits with trackId +
#     eventId. Adds the merged simHit cost (calo, plus tracking under
#     includeTrackingHits).

import FWCore.ParameterSet.Config as cms

# The compact, stage-independent truth: the physics graph, the unresolved
# per-particle per-cell footprint, and the raw merged CSR graph (needed by the
# branch validators' rawSrc and for offline rebuild; ~3 MB/ev).
_truthGraphKeep = [
    'keep *_truthLogicalGraphProducer_*_*',
    'keep *_truthLogicalGraphHitIndexProducer_*_*',
    'keep TruthGraph_mix_*_*',
]


def _truthSimHitsKeep(includeTrackingHits):
    """The merged (signal+pileup) simHits kept only at the 'full' level, so the
    numerator/denominator can be rebuilt at a different granularity. Calo always;
    tracking (tracker + muon + MTD) only when the accumulator captured it."""
    keep = [
        'keep *_mix_mergedHGCHits_*',
        'keep *_mix_mergedEcalHits_*',
        'keep *_mix_mergedHcalHits_*',
    ]
    if includeTrackingHits:
        keep += [
            'keep *_mix_mergedTrackerHits_*',
            'keep *_mix_mergedMuonHits_*',
            'keep *_mix_mergedMtdHits_*',
        ]
    return keep


# The default analysis-tier content.
truthContentCompact = cms.untracked.vstring(_truthGraphKeep)


def truthContentFull(includeTrackingHits=True):
    """The 'full' content: compact + merged simHits."""
    return cms.untracked.vstring(_truthGraphKeep + _truthSimHitsKeep(includeTrackingHits))


def truthEventContent(level='compact', includeTrackingHits=True):
    """Return the output commands for a truth verbosity level.
    level='compact' (default) or 'full'."""
    if level == 'compact':
        return cms.untracked.vstring(truthContentCompact)
    if level == 'full':
        return truthContentFull(includeTrackingHits)
    raise ValueError("unknown truth event-content level %r; choose 'compact' or 'full'" % level)


def setTruthEventContent(process, level='compact', includeTrackingHits=True):
    """Append a truth verbosity level to every output module in the process.
    level='compact' (default, ~O(10) MB/ev: graph + unresolved hit index) or
    'full' (adds raw graph + merged simHits for off-geometry re-association)."""
    commands = truthEventContent(level, includeTrackingHits)
    for out in process.outputModules_().values():
        out.outputCommands.extend(commands)
    return process
