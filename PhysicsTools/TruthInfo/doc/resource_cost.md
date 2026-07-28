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

Everything here is NO PILEUP. PU200 is not measured and the ratios are expected to move
there, because the legacy objects and the graph scale differently with the number of
overlaid interactions.

## 1. Event size

Read with `edmEventSize -v`. Both schemes first appear at DIGI and are copied unchanged
into RECO, so the DIGI and RECO byte counts are identical.

| Scheme | branches | uncompressed kB/event | compressed kB/event |
|---|---:|---:|---:|
| Legacy: TrackingParticle, TrackingVertex, CaloParticle, SimCluster and their Refs | 14 | 1731.4 | 622.5 |
| Graph: `TruthGraph`, `truth::Graph`, `truth::LogicalGraphHitIndex` | 3 | 1564.3 | 415.4 |
| Truth-branch association maps, 3 domains x 4 working points x 2 directions | 27 | 172.2 | 29.2 |
| **Graph total** | **30** | **1736.5** | **444.6** |

Dropping the legacy collections and keeping the graph plus all four association working
points saves **177.9 kB/event compressed, 29%** of the truth payload. The four working
points are a validation convenience: a production configuration with one working point
would write about 7 kB/event of maps instead of 29, so the saving becomes about 32%.

The single largest graph branch is the hit index at 271.0 kB/event compressed. It is a
separate product and can be dropped on its own; the two graph structures alone are
144.4 kB/event.

For context, the whole RECO event is 8043.0 kB/event compressed. Graph products are 5.5%
of it, legacy truth 7.7%.

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

## 5. Not measured

- PU200, or any pileup at all. The DIGI log of this chain even warns that pileup-aware
  truth needs classic, non-premixed pileup.
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
associators is a **29% reduction of the persisted truth payload** (622.5 to
444.6 kB/event compressed, or about 32% with a single working point), a **factor 2.9 less
memory allocated during mixing** (29.1 to 9.9 MB/event), and a RECO-side association cost
of 4.28 ms/event, 0.38% of the scheduled reconstruction. The DIGI-time accumulation step
itself is not removed, and the pileup case is not yet measured.
