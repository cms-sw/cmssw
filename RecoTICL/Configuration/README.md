<!-- Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch -->
# pyTICL

A validated, type-safe configuration framework for TICL (the Phase-2 CMS HGCAL
reconstruction).  Instead of editing large hand-written `_cff.py` fragments,
declare a configuration with a concise fluent builder; pyTICL builds the `cms`
modules, **checks the plumbing with type-aware rules**, lets you pick CPU/GPU per
module, and **exports a standalone cff** (useful for HLT).

## Quick start

```python
from RecoTICL.Configuration import presets

cfg = presets.v5()           # the default iterTICLTask, as a TICLConfig
cfg.validate()               # type-aware plumbing checks (raises on problems)
cfg.to_cff('myTICL_cff.py')  # write a self-contained, loadable cff fragment
```

Building one by hand with the fluent API:

```python
from RecoTICL.Configuration.model import TICLConfig, Global
from RecoTICL.Configuration import presets

cfg = (TICLConfig('v5')
       .iteration('CLUE3DHigh')
           .seeding(Global)
           .filter_by_algo_and_size(min_size=2)
           .pattern_clue3d(criticalDensity=[0.6, 0.6, 0.6],
                           criticalEtaPhiDistance=[0.025, 0.025, 0.025])
       .iteration('Recovery').preset()        # standard Recovery fragment
           .masks_from('CLUE3DHigh')
       .links(['CLUE3DHigh', 'Recovery'], **presets.links_defaults())
       .superclustering_dnn(source='CLUE3DHigh', **presets.supercluster_dnn_defaults())
       .candidate(**presets.candidate_defaults())
       .pf(**presets.pf_defaults()))

cfg.validate()
```

`assemble()` returns the built modules + tasks; `add_to_process(process)`
registers them on a `cms.Process`.

### Targets (offline / HLT)

The same declaration can be emitted for a different **target**.  A `Target`
(see `target.py`) captures what differs between offline reco and the Phase-2
HLT: the module label scheme (`ticlTracksters…` vs `hltTiclTracksters…`), the
merged layer-cluster source (`hgcalMergeLayerClusters` vs `hltMergeLayerClusters`)
and the inputs derived from it, and whether stages are grouped into `cms.Task`
(offline) or `cms.Sequence` (HLT).

```python
from RecoTICL.Configuration import hlt_presets
cfg = hlt_presets.v5_hlt()    # reproduces HLTIterTICLSequence (9 modules)
cfg.to_cff('hltTICL_cff.py')  # emits hlt-prefixed cms.Sequences
```

Note: the `HLT_75e33` menu is a frozen confdb-style dump that can lag the live
producers.  pyTICL clones the *live* `_cfi` defaults, so it reproduces the HLT
**structure and plumbing exactly** and *reports* the residual parameter deltas as
"frozen-menu drift" (parameters the producers gained after the menu was frozen) --
a signal that the menu should be regenerated.

## How it works

| Module | Responsibility |
| --- | --- |
| `catalog.py` | type-aware registry: each producer's `consumes`/`produces` C++ types, instance-label rules, backends, and the valid plugin-type strings |
| `model.py` | the fluent builder (`TICLConfig`); records intent only |
| `presets.py` | the standard v5 iterations & singletons (algorithm PSets transcribed from the baseline; plumbing left to pyTICL) |
| `assembler.py` | clones the real `_cfi` defaults + applies algorithm overrides and the **computed** plumbing `InputTag`s; builds the `Task` hierarchy |
| `validator.py` | builds the product graph and rejects type-incompatible / missing / GPU-unsupported connections |
| `exporter.py` | `to_cff(path)` -- a self-contained, loadable cff fragment |
| `validation.py` | single-source-of-truth derivation of the iteration labels, associator instances, the (auto-created & scheduled) trackster<->simTrackster associators, the `ticlDumper` and the `hgcalValidator` |
| `backend.py` | CPU/GPU (alpaka) selection: drive a portable `@alpaka` module to `serial_sync`/`cuda_async` and wire up `ProcessAcceleratorAlpaka`; GPU on a CPU-only module is rejected |
| `compare.py` | per-module comparison of an assembled config vs a baseline `Task` |

From one list of labels, `cfg.build_validation()` derives the whole validation
chain exactly as the baseline does (verified byte-for-byte for the associators
and `hgcalValidator`), so reco, dumper and validation never drift apart.

Byte-for-byte reproduction is achievable because the assembler clones the *same*
`_cfi` defaults the baseline uses and re-applies the same overrides; the
framework's own contribution is the wiring, validation, backend selection, and
export.

## Validation & drift detection (`test/`)

* `test_reproduce_v5.py` -- **acceptance gate**: the generated v5 config equals
  the live `iterTICLTask` byte-for-byte.  Also the primary drift detector: if a
  baseline cff changes in a way pyTICL doesn't mirror, this fails with a diff.
* `test_catalog_schema.py` -- locks the catalog against the live `_cfi` defaults;
  fails if a producer gains/loses/retypes an `InputTag` parameter (new plumbing
  pyTICL must learn about).
* `test_plumbing.py` -- positive + negative validator tests (e.g. GPU on a
  CPU-only module is rejected).
* `test_export_cff.py` -- the exported cff reloads and reproduces the baseline.

Run them with `scram b runtests` (from the package) or directly with `python3`.

## Status / roadmap

Implemented: the v5 TICL core (iterations, links, superclustering, candidate,
pf), type-aware validation, cff export, drift tests, and the **Phase-2 HLT
target** (reproduces `HLTIterTICLSequence`).

Planned (see the package design notes): auto-derived & scheduled
labels/validation/dumper/associators; local reconstruction + layer clustering
(HGCAL/ECAL/HCAL) with real CPU/GPU (alpaka) backend selection; the TICL barrel
path.
