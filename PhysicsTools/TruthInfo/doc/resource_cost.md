# Computing cost: the MC-truth graph against the legacy frozen truth objects

What it costs to keep `TrackingParticle`, `TrackingVertex`, `CaloParticle` and
`SimCluster`, and what it costs to keep the truth graph plus its associators instead.
Every number below was measured on one sample; nothing is extrapolated. Where a
quantity was not measured it is listed as not measured rather than estimated.

## Sample and conditions

Identical for every number in this document.

| | |
|---|---|
| Process | `TTbar_14TeV_TuneCP5`, no pileup (`mixNoPU_cfi`) |
| Events | 10 |
| Release | `CMSSW_20_1_X_2026-07-22-2300` |
| Geometry, era, global tag | `ExtendedRun4D122`, `Phase2C26I13M9`, `auto:phase2_realistic_T35` |
| Steps | GEN,SIM then DIGI,L1,DIGI2RAW,HLT:@relvalRun4 then RAW2DIGI,L1Reco,RECO |
| Host | AMD EPYC 9754, 1 thread, 1 stream |

Sections 1 to 6 are NO PILEUP. Section 7 measures the event size at PU200, where the
two schemes scale very differently; the CPU and memory numbers remain no-PU only.

## 1. Event size

Read with `edmEventSize -v`. Both schemes first appear at DIGI and are copied unchanged
into RECO, so the DIGI and RECO byte counts are identical.

| Scheme | branches | uncompressed kB/event | compressed kB/event |
|---|---:|---:|---:|
| Legacy: TrackingParticle, TrackingVertex, CaloParticle, SimCluster and their Refs | 14 | 1731.4 | 622.5 |
| Graph, **default** (shared) hit index: `TruthGraph`, `truth::Graph`, `truth::LogicalGraphHitIndex` | 3 | 1149.8 | 368.3 |
| Graph, materialised hit index (`sharedSubgraphStore=False`) | 3 | 3304.2 | 611.3 |
| Truth-branch association maps and their root/target index vectors | 23 | 168.8 | 25.1 |
| **Graph total, default index** | **26** | **1318.6** | **393.4** |

The legacy row is carried over unchanged: nothing in the graph work touches those
collections. The graph and association rows were re-measured on the current build.

Dropping the legacy collections and keeping the graph plus all four association working
points saves **229.1 kB/event compressed, 37%** of the truth payload, and that is with the
full signal GEN half included, which the legacy collections do not carry at all. Keeping
the materialised index instead would cost 13.9 kB/event more than legacy, +2.2%, so the
shared index is what makes the graph a saving rather than a cost.

!!! note "The chain is not bit-reproducible, so read the last digit as noise"
    Two runs of the identical configuration differ in 64 of 1338 branches, HLT tracking
    among them. `TruthGraph_mix` ranged 53793 to 56124 compressed bytes/event over four
    runs, a 4.2% spread, so the graph row uses its mean of 54.9 kB/event. The hit index
    and `truth::Graph` are stable to the byte, and so is everything derived from them.
    Section 6 has the detail.

The single largest graph branch is the hit index: 202.3 kB/event compressed by default,
445.3 if the materialised layout is selected. It is a separate product and can be dropped on its own; the
two graph structures alone are 166.0 kB/event.

For context, the whole RECO event is 7991.4 kB/event compressed. Graph products are 4.9%
of it, legacy truth 7.8%.

### 1.1 The table above predates the pruning-scope fix and is now an UNDERCOUNT

The hitless-subgraph pruning used to be wired to the HGCAL endcap only, so every
particle depositing solely in the ECAL/HCAL barrel, and under pileup every pileup
charged particle, was pruned as hitless. The graph was smaller because it was wrong.

Controlled A/B, same 10 ttbar events, same job, only the producer's sim-hit collections
changed (compressed bytes per event):

| branch | pruning on HGCAL only | pruning on the full detector |
|---|---:|---:|
| `TruthGraph` (`mix`) | 35264 | 34661 |
| `truth::Graph` | 50570 | 128166 |
| `truth::LogicalGraphHitIndex` | 116202 | 143477 |
| **total** | **202037** | **306304** |

The graph branch grows 2.5x because it now keeps the barrel particles it always should
have kept. The table in section 1 was measured before this fix and must be re-measured
on the reference conditions before any size claim is quoted from it again.

### 1.2 What changed since the first version of this document

Two things moved in opposite directions and the net is a saving.

The graph now carries the **full signal GEN half**, contracted: the parton shower and the
intermediate copies of a resonance are collapsed away by `truth::collapseGenShower`, so a
resonance appearing several times is one node whose children are its decay products. That
half did not exist when this document was first written.

The hit index now uses the **shared subgraph store**. Each hit is written once, in an
order that makes a particle's subtree a contiguous run of slots, so a subgraph is a set of
ranges of that one store rather than a second materialised copy under every ancestor. A
GEN-only particle sits above the SIM tree in a DAG, so it owns a merged set of runs rather
than a single one; measured over 3 events, 1039 such particles need 2547 ranges, mean
2.45, median 1, max 183.

A/B on the same GEN-SIM, 10 events, compressed bytes/event of the three graph branches:

| variant | hit index | `truth::Graph` | `TruthGraph_mix` | total |
|---|---:|---:|---:|---:|
| no GEN half, materialised index | 271013 | 92499 | 54037 | 417549 |
| full GEN half, materialised index | 696729 | 126768 | 74681 | 898178 |
| collapsed GEN half, materialised index | 445329 | 111070 | 55413 | 611812 |
| **collapsed GEN half, shared index (the default)** | **202300** | **111070** | **54938** | **368308** |

The `TruthGraph_mix` column carries the run-to-run spread noted above, so differences of
about a kB in that column and in the total are not significant. The hit index column,
which is what these variants are about, is stable to the byte.

The materialised index stored a hit once per ancestor containing it: 26744 hits per event
became 46259 stored entries even with no GEN half at all, and 1467228 with the full one.
The shared store keeps the 26744. Reading is automatic in both layouts, which is what
keeps every file written before this change readable.

The cost is paid at query time: a range is in tree order and repeats a detId hit by
several descendants, so a consumer that needs per-cell energies coalesces it.
`BranchHitAssociator` does this once per candidate root at construction. Measured on the
same 10 events, `allTrackToTruthBranchAssociators` goes from 10.1 to 12.2 ms/event and the
whole RECO job from 692.5 to 694.9 ms/event, +0.3%.

### Cost per object

| Collection | objects/event | compressed bytes/object |
|---|---:|---:|
| `truth::Graph` particles + vertices | 1186.5 + 557.3 | 53 |
| `TrackingParticle` (MergedTrackTruth) | 896.1 | 99 |
| `CaloParticle` (MergedCaloTruth) | 185.4 | 414 |

The graph node is cheaper for a structural reason, not a compression accident. A
`TrackingParticle` embeds `std::vector<SimTrack> g4Tracks_`, and so do `CaloParticle` and
`SimCluster`; each `SimTrack` carries 12 members and the same SimTracks are already
persisted separately in `SimTracks_g4SimHits` (431.2 kB/event compressed). The legacy
classes also store their topology as `edm::Ref` and `RefVector` members
(`parentVertex_`, `decayVertices_`, `daughterTracks_`, `sourceTracks_`, `simClusters_`).
`truth::ParticleData` and `truth::VertexData` embed no SimTrack and no Ref: topology
lives once, in the CSR offset arrays of `truth::Graph`, and the hit payload is factored
out into the hit index instead of being duplicated as parallel `hits_` and `fractions_`
vectors in every `CaloParticle` and every `SimCluster`.

The duplication is the point. The legacy scheme writes four distinct `SimCluster`
collections plus `CaloParticle` plus `TrackingParticle` and `TrackingVertex`, each
re-embedding its own SimTrack copies and hit arrays. The graph writes one structure
covering tracker and calorimeter together: 1186.5 particles per event against
896.1 + 185.4 + 423.5 legacy objects.

## 2. CPU and allocated memory at DIGI

`TrackingTruthAccumulator`, `CaloTruthAccumulator` and `TruthGraphAccumulator` are all
`DigiAccumulatorMixMod` plugins inside the single `mix` module, so the FastTimerService
cannot separate them. They were isolated by an A/B on the same input, same 10 events,
same host, deleting entries from `process.mix.digitizers`, three repetitions each.

| Variant | mix real ms/event (mean, sd, n=3) | mix allocated kB/event |
|---|---:|---:|
| as shipped | 2449.0, 2.6 | 969 965.4 |
| without `TruthGraphAccumulator` | 2469.1, 40.6 | 960 045.1 |
| without `TrackingTruth` and `CaloTruth` | 2430.0, 14.2 | 940 884.7 |
| without all three | 2408.2, 6.2 | 930 964.4 |

All three truth accumulators together cost **40.8 +- 3.9 ms/event**, which is **1.7% of
the 2449 ms/event that `mix` costs**. The split of that time between graph and legacy is
NOT resolved at three repetitions: the two independent estimates of each disagree by
more than their errors.

The allocated-memory split is exact and additive (9920.3 + 29080.7 = 39001.0):

- `TruthGraphAccumulator`: **9.9 MB/event**
- `TrackingTruthAccumulator` + `CaloTruthAccumulator`: **29.1 MB/event**, a factor **2.9**
  more.

The graph's own downstream stages are ordinary EDProducers and are separately timed:
`truthLogicalGraphProducer` 3.5 ms/event, `truthLogicalGraphHitIndexProducer`
5.4 ms/event.

## 3. CPU at RECO

Re-run of the RECO step with the FastTimerService; the framework TimeReport agrees to the
microsecond.

| Module | real ms/event | allocated kB/event |
|---|---:|---:|
| `allTrackToTruthBranchAssociators` | 3.713 | 1669.0 |
| `allVertexToTruthBranchAssociators` | 0.475 | 155.6 |
| `allSecondaryVertexToTruthBranchAssociators` | 0.091 | 140.3 |
| total | 4.28 | 1964.9 |

That is **0.38% of the 1119.7 ms/event** summed over all scheduled RECO modules, and it
is with four working points per domain. No legacy truth producer runs in this RECO
sequence, so there is no legacy counterpart to compare the associators against.

## 4. What the graph does NOT save

Stated plainly, because the opposite is easy to assume.

**The DIGI-time accumulation step is not removed.** All three accumulators exist for the
same reason: pileup sub-event SimTracks, SimVertices and SimHits are only reachable
through `PileUpEventPrincipal` while mixing runs, and are never in the output event. The
sources make this explicit:

- `SimGeneral/TrackingAnalysis/plugins/TrackingTruthAccumulator.cc:407` and `:471`
- `SimGeneral/CaloAnalysis/plugins/CaloTruthAccumulator.cc:684` and `:782`
- `PhysicsTools/TruthInfo/plugins/TruthGraphAccumulator.cc:458` and `:469`

The graph replaces two accumulators with one; it does not eliminate the step. The saving
is in what that step allocates and writes, not in skipping it.

**MTD legacy truth is untouched.** `MtdSimClusters`, `MtdSimLayerClusters`,
`MtdSimTracksters` and `MtdCaloParticles` are a further 243.6 kB/event compressed that
the graph does not currently replace.

**Raw SimTracks and SimVertices stay.** 529.8 kB/event compressed, written by the SIM
step and consumed by both schemes.

## 5. Reading a subgraph in either layout

`LogicalGraphHitIndex::subgraphHits` returns a single span. In the shared layout a
particle that carries hits owns exactly one slot range, so its span is fine, but a
GEN-only particle owns several and the accessor returns an **empty** span. Four validators
used its size as a smallest-footprint tie-break, where a zero-size answer makes a GEN-only
root win a comparison meant to pick the tightest branch, so a naive switch of layout would
have read as a near-total loss of reproduction efficiency.

`truth::SubgraphHitView` is what consumers use instead. It returns the coalesced,
detId-sorted span in either layout: the materialised one already persists that form and is
handed back untouched, the shared one is coalesced once per particle and cached, and a
particle whose subgraph is a single one-slot range needs neither, since the builder already
sorted and summed its own direct hits. Hold one per event and per module; it caches, so it
is not thread safe.

All seven consumers now go through it, and the arithmetic in each is unchanged. Verified on
10 ttbar events by running the calorimeter validator over a materialised index and a shared
index and comparing every monitor element: **50 compared, 50 non-empty, 0 differing**. That
rests on the accessor-level equivalence measured in section 1.1, which covered every
particle and every channel.

## 6. This chain is not bit-reproducible, and that is not a truth-graph property

Read the byte counts in section 1 with this in mind. Two runs of the identical
configuration, same input file, same host, one thread, do not write the same file:
**64 of 1338 branches differ in compressed size**, and five of them differ uncompressed
too, so the difference is real content and not just packing. The largest by absolute
delta happens to be `TruthGraph_mix` (56123.9 against 55250.2 compressed bytes/event,
uncompressed identical at 498985), but it is in company:

| branch | compressed | uncompressed |
|---|---|---|
| `TruthGraph_mix` | 56123.9 -> 55250.2 | same |
| `recoTrackExtras_hltGeneralTracks` | 46859.7 -> 46867.5 | differs |
| `recoTracks_hltInitialStepTrackSelectionHighPurity` | 8934.8 -> 8937.2 | same |
| `recoVertexs_hltOfflinePrimaryVertices` | 521.2 -> 519.8 | same |
| `uints_hltPhase2PixelTracksCAExtension` | 525.9 -> 527.2 | same |

HLT tracking output changing between identical runs is the root of most of this, and it
is upstream of and independent of the truth graph.

The part that matters here is that the truth **content** is reproducible. `truth::Graph`
and `truth::LogicalGraphHitIndex` hash identically across the same runs, over the logical
particle and vertex records, all eight CSR arrays, and every hit and offset of all four
channels. Of `TruthGraph`'s own persisted arrays, `offsets`, `edges`, `edgeKind`,
`pdgId`, `status`, `statusFlags`, `eventId`, `genEventOfNode`, `simVertexProcessType`,
`simTrackBackscattered` and `simTrackToGen` are all identical run to run.

One loose end specific to this package, worth a look but not a blocker:
`TruthGraph_mix` is the one branch whose compressed size moves while its uncompressed
size does not, which means same length and different bytes. `TruthGraph::NodeRef` is
`{ NodeKind kind; int64_t key; }` with `NodeKind` a `uint8_t`, so it carries seven bytes
of padding and the branch is written unsplit; uninitialised padding would produce exactly
that signature without changing any value a consumer reads. This is a hypothesis, not a
result: confirming it needs a C++-side dump of the raw buffer, because PyROOT cannot read
the unsplit struct reliably, which is also why `kind` is excluded from the hashes above.

## 7. Event size at PU200

Same signal process, same release and geometry, 10 events, classic mixing with an
average of 200 minimum-bias interactions from a D122 truth-enabled library, default
truth wiring, no selection preset. Compressed kB/event:

| Scheme | kB/event | vs no pileup |
|---|---:|---:|
| Legacy: TrackingParticle, 2x TrackingVertex, 4x SimCluster, CaloParticle | 56717 | x91 |
| Graph: `TruthGraph`, `truth::Graph`, `truth::LogicalGraphHitIndex` | 8791 | x24 |

At PU200 the graph is **15.5% of the legacy truth payload**, a factor 6.5, against 59%
with no pileup: the saving grows with pileup because the legacy objects re-embed their
SimTrack copies and hit arrays per object and per collection, and pileup multiplies the
objects, while the graph's topology stays CSR and its hits stay stored once. The shared
hit index is 4762 kB/event of the graph total and remained the persisted layout in 10 of
10 events, with no fallback and no degradation warnings.

MTD legacy truth, which neither scheme replaces, is a further 9859 kB/event at PU200.

The pileup GEN half is collapsed (`collapsePileupGen=True`): each pileup interaction
carries one Interaction vertex, one UnderlyingEvent vertex holding its stable particles,
and nothing else, so the graph cost of 200 extra interactions is dominated by their
SIM tracks and hits, not their generator records.

## 8. Not measured

- CPU and allocated memory at PU200. Section 7 measures the event size there; the
  accumulator A/B of section 2 and the RECO timings of section 3 are no-PU only.
- The RECO-side associator numbers in section 3 predate the shared hit index. Section 1.1
  gives the measured before/after for `allTrackToTruthBranchAssociators` on the same 10
  events, but the per-module table in section 3 was not re-measured with the
  FastTimerService.
- Peak RSS. No log in this chain reports `SimpleMemoryCheck`, and no instrumentation was
  added. The memory figures above are the FastTimerService allocated-bytes counter.
- The CPU split between `TruthGraphAccumulator` and the two legacy accumulators inside
  `mix`. Only the combined 40.8 +- 3.9 ms/event is resolved at three repetitions.
- A true A/B of file size with the graph removed from the chain; the per-branch sizes
  above are read off the single existing chain.
- Legacy associator CPU at RECO, for example `quickTrackAssociatorByHits`, which is not
  in this RECO sequence.
- Any multi-threaded scaling. Everything is one thread, one stream.

## Summary

On a no-pileup ttbar event, replacing the frozen truth objects with the graph and its
associators is a **37% reduction of the persisted truth payload** (622.5 to
393.4 kB/event compressed), a **factor 2.9 less memory allocated during mixing** (29.1 to
9.9 MB/event), and a RECO-side association cost of a few ms/event, well under 1% of the
scheduled reconstruction. That is with the full signal GEN half included, which the
legacy collections do not carry at all. The DIGI-time accumulation step itself is not
removed. At PU200 the size advantage grows to a factor 6.5 (section 7); the CPU and
memory numbers remain no-PU only.
