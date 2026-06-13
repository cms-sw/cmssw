# `truth::Branch` — design proposal (for discussion)

## Concept
A `truth::Branch` is a **coherent connected subgraph** of `truth::Graph`: a chosen
root (a particle, or a small set such as `Z -> mu mu`) together with a defined
**closure** of its descendants, plus the detector footprint attached through
`LogicalGraphHitIndex`. Like `Particle`/`Vertex`, it is a lightweight **view**
into a `Graph`, not an owning copy.

A Branch is the natural target for truth-reco association when a reconstructed
object does not map to a single truth particle:
- a jet  <-> a parton's whole shower branch,
- an ECAL supercluster <-> a photon + its conversion e+e- branch,
- a tau-jet <-> the visible tau-decay branch,
- a b-jet  <-> the b-quark branch (with its B-hadron sub-branch).

It is built directly on phases 1-3: the selection (seed PDG / heavy-flavor)
chooses the root, the downstream closure defines the extent, the ISR/underlying
-event roles + genEvent/eventId give provenance, and the hit index gives the
detector footprint.

## Construction & closure policy
```cpp
Branch b = graph.branch(rootParticle, closure);
```
`closure` selects which members belong to the branch and is always evaluated on
the fly from the `Graph` (a Branch is never an EDM product):
- `Subtree`     - the root and all descendants (default);
- `StableLeaves`- the root plus only its final-state descendants;
- `DepthN`      - descendants down to N generations;
- `UntilPdgId`  - stop the closure at a species (e.g. stop at stable hadrons,
                  or at the B hadron for a b-branch);
- `Predicate`   - stop on a user predicate (stop at any heavy-flavor hadron, at a
                  detector-boundary crossing, on a custom lambda), so closures
                  are extensible without new enum values.
The phase-1-3 postprocessing already computes member sets; a Branch makes that
set a first-class, queryable object.

**Decision:** the Branch is a **view, recomputed on demand** — stateless, no
stored member list, not an EDM product. Any caching needed for performance lives
in the *matching layer* (below), scoped to a batch of objects, not in the Branch.

## Data model
```cpp
class Branch {
  Graph const* graph_;
  std::vector<uint32_t> roots_;      // usually 1
  std::vector<uint32_t> members_;    // closure (materialized)
  // optional caches: p4 sums, DetId set, hit spans
};
```
A Branch carries provenance via its root (`genEvent`/`eventId`), so pile-up
branches stay distinguishable when graphs are overlaid.

## Queries the Branch should answer

### A. Matching reco objects (the substrate is detector-agnostic; metrics are pluggable)
- `members()`, `stableLeaves()`, `chargedStableLeaves()`.
- `hits(closure)` - aggregated direct/subgraph SimHits + matched RecHits over all members (LogicalGraphHitIndex already gives this per particle).
- `detIds()`, `energy(Detector)` - sim/rec energy summed over the branch in a subdetector.
- `sharedHitFraction(recoObject)` / `sharedHits(recoObject)` - tracking-style metric.
- `energyFraction(recoCluster)` - calorimeter-style metric.
- `matchScore(recoObject, Metric)` - **Metric is a strategy on the Branch**: shared-hits (tracking), energy-fraction (calo), time-aware (MTD). New detectors add a Metric without touching Branch.
- `containsSimTrack(id)`, `containsDetId(id)`.

**Batch / many-to-many matching.** Single-object queries go through the Branch
metric strategy above. For associating *collections* — N reco <-> 1 sim (split
tracks, calo fragments -> one particle) and N sim <-> 1 reco (a jet <- a branch)
— a free `BranchMatcher(branches, recoObjects, Metric)` builds the inverted
`hit/DetId -> branch` index **once and caches it for the duration of the call**,
then emits a weighted bipartite association in both directions (cf. reco's
`RecoToSimCollection`/`SimToRecoCollection`). The Branch stays stateless; the
cache lives in the matcher.

**Hit ranges and `std::span`.** If `LogicalGraphHitIndex` lays hits out in graph
-topological order, a `Subtree` branch's hits are a **contiguous range** — i.e.
exactly the precomputed subgraph-hit `std::span` of its root, returned with zero
gather. The matcher can then count shared hits / energy by a sorted-range
merge-join rather than hashing, which is the cache-friendly path. (This needs the
hit-index builder to guarantee the topological layout; see follow-up below.)

### B. Tagging (flavor / origin / process)
- `rootPdgId()`, `originPdgId(targets)` (= `firstAncestorWithPdgId` from the root).
- `hasAncestorPdgId(id)` - is this branch from a top? a Z? the hard scatter?
- `heavyFlavorContent()` - does the branch contain a b/c hadron (b/c-tag truth)? (reuses the phase-2 flavor classifier).
- `isFromHardScatter()` / `isFromPileup()` - via the genEvent/eventId provenance (phase 1).
- `decayChannel()` - decay mode of the root (Z->mumu vs Z->ee, tau hadronic vs leptonic, prong count).
- `flightLength()` / `displacedVertex()` - production->decay displacement (Lxy) for b/tau lifetime tagging.

### C. Physics performance
- `p4(MemberSelector)` - branch four-momentum over {all | stable | charged | visible}.
- `visibleEnergy()` / `invisibleEnergy()` - missing energy from neutrinos/LSP.
- `chargedFraction()`, `emFraction()`, `hadronicFraction()` - for jet response/composition.
- `response(recoObject)` = recoE / branchE; efficiency/fake bookkeeping via matching.
- boundary-crossing kinematics from `Checkpoint`s - for propagation/calibration studies.

### D. Branch <-> Branch relations
- `commonAncestor(other)` - do two branches come from the same top / same Z? (generalizes `lowestCommonAncestor`).
- `merged(other)` - combine two branches (and their hit content) into one (e.g. the two Z-decay legs).
- `deltaR(other)`, `overlap(other)` (shared members/hits) - for splitting/merging studies.

## Cross-cutting principles
- **Substrate vs metric**: the Branch holds structure + hits; matching *definitions*
  stay detector-aware and use-case dependent (tracking != calo != timing).
- **Provenance-aware**: every Branch knows its source event (primary vs pile-up).
- **Composable**: branches merge/split; queries compose with the existing
  navigation (`ancestors`, `firstCommonAncestor`, `firstAncestorWithPdgId`).
- **Built on what exists**: selection (phase 1-2), navigation (phase 3),
  hit index (existing) — Branch is the unifying view, not new infrastructure.

## Resolved decisions
1. **View, recomputed on demand** — the Branch stores no member list and is not
   an EDM product.
2. **Derived on the fly** from `Graph` + closure (never persisted). Member-id
   lists are cheap to recompute; hit aggregates are not stored on the Branch.
3. **Metric strategy on the Branch** for single-object scores; a free,
   cache-holding `BranchMatcher` for batch many-to-many association.
4. **`merged()` unions member sets lazily** (no cached hit aggregates),
   consistent with the view model.
5. **Closures include predicate-based stops**, in addition to the fixed
   `{Subtree, StableLeaves, DepthN, UntilPdgId}` set.

## Follow-ups
- Guarantee a graph-topological hit layout in `LogicalGraphHitIndexBuilder` so a
  `Subtree` branch's hits are a contiguous `std::span` (zero gather; merge-join
  matching).
- Choose the first concrete `Metric`s (shared-hits, energy-fraction) and the
  first reco target (tracks, then jets) when implementation starts.
