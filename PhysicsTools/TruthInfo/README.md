# TruthInfo prototype

> **Status: under heavy development — not open to external contributions.**
> This is an experimental prototype: its data model, APIs, and configuration
> change frequently and without notice, and it targets **Phase-2 (Run 4) only**
> (no Phase-1/Run-2 support). Please do not submit external changes or depend on
> it in production at this stage.
>
> **Original author and maintainer:** Felice Pantaleo (CERN),
> <felice.pantaleo@cern.ch>.

A prototype **MC-truth graph** for CMS: a single, navigable, physics-oriented
abstraction of the generator + simulation truth history of an event, with
calorimeter and tracker hit indices layered on top. It replaces the need to
cross-navigate the many low-level truth collections (HepMC, GenParticles,
SimTracks/SimVertices, TrackingParticles, SimClusters, CaloParticles, SimHits,
RecHits) by hand.

## Documentation

The **authoritative, maintained documentation** lives on the project website:

> **http://cms-truth.docs.cern.ch/**

It covers the data model, how to enable and use the graph (navigation API, the
`truth::Branch` view, the hit index, matching reco objects), the physics findings,
validation, pileup, and the roadmap. The MkDocs sources are in the companion
`cms-truth-docs` repository. Start with the **"How to use the graph"** page.

## The three layers

1. **`TruthGraph`** (raw) — a compact CSR graph built directly from HepMC +
   `SimTrack`/`SimVertex` by `TruthGraphProducer`.
2. **`truth::Graph`** (logical) — a user-facing bipartite Particle↔Vertex graph
   built by `TruthLogicalGraphProducer`; GEN and SIM are merged where robustly
   associated, with navigation (`parents()`, `descendants()`,
   `firstCommonAncestor()`, `hasAncestorPdgId()`, …) and the `truth::Branch`
   subgraph view + `BranchSelector` selection.
3. **`truth::LogicalGraphHitIndex`** — per-particle direct vs aggregated subgraph
   calorimeter and tracker hits, built by `LogicalGraphHitIndexProducer` (with the
   DetId→RecHit map from `DetIdToRecHitMapProducer`).

Producer chain (order matters): `truthGraphProducer` → `truthLogicalGraphProducer`
→ `detIdToRecHitMapProducer` → `truthLogicalGraphHitIndexProducer`. In a release
job these run behind the `enableTruth` process modifier (the
`truthGraphPrevalidation` sequence in `Validation/Configuration`).

## Package layout

- `interface/` + `src/` — the data formats and algorithms: `TruthGraph`,
  `truth::Graph`, `Branch`, `BranchSelector`, `LogicalGraphHitIndex`,
  `BranchHitAssociator`, the `truth::recoHits` adapters (`RecoHitAdapters.h`), and
  `TruthLogicalGraphPostProcessor` (merge/collapse/filter; covered by the cppunit).
- `plugins/` — the producers above, the DOT dumpers, the flat-table producers, the
  pileup `TruthGraphAccumulator`/`TruthGraphMixedProducer`, the association-map
  producers (`TruthBranchCaloAssociationProducer`,
  `TruthBranchTrackingAssociationProducer`), and the DQM validators
  (`BranchHGCalValidator`, `BranchTrackingValidator`, the generic
  `BranchRecoValidator`).
- `python/` — `truthGraphValidation_cff` (producers + association maps + DQM
  analyzers) and `truthGraphDQMHarvester_cff` (efficiency/fake/merge harvesting).
- `scripts/` — `makeTruthGraphValidationPlots.py` (renders the Branch validation
  plots / sample overlays).
- `test/` — cppunit unit tests and standalone `cmsRun` drivers (graph dumps,
  topology checks, association/DQM smoke tests).

## Build, check, test

All commands assume the CMSSW environment (`cd $CMSSW_BASE/src && cmsenv`):

```bash
scram b -j 8                          # build
scram b code-format code-checks -j 8  # clang-format + clang-tidy (must pass)
scram b runtests                      # cppunit unit tests
```

## Run standalone on a step3.root

```bash
# Dump per-event DOT graphs (options: -n, -m/--merge, -c/--collapse, -o, -t)
cmsRun test/dumpTruthGraphsFromGENSIMRECO_cfg.py path/to/step3.root -n 5

# Branch DQM validators (calo / tracking / generic reco-side)
cmsRun test/validateBranchDQM_cfg.py          path/to/step3.root -n 5
cmsRun test/validateBranchTrackingDQM_cfg.py  path/to/step3.root -n 5
cmsRun test/validateBranchRecoDQM_cfg.py      path/to/step3.root -n 5
```

See the website for the full configuration reference, the navigation API with
examples, and the validation results.
