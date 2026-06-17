# TruthInfo test & debugging tools

Reusable helpers for producing and inspecting truth graphs. All require `cmsenv`
(run from `CMSSW_17_0_0_pre2/src`).

| Tool | Purpose |
|---|---|
| `dumpTruthGraphsFromGENSIMRECO_cfg.py` | cmsRun config: build the raw + logical truth graph from a GEN-SIM/RECO file and dump DOT (+NanoAOD rechit/simhit tables). Selection flags: `-s/--seeds`, `-g/--groups`, `-d/--parentDepth`, `-i/--ignore`, `-m/--merge`, `-c/--collapse`, `--showAll`, `-n`, `-o`, `-t`. `-s 0` keeps the full graph. |
| `truthGraphConnectivity.py` | FWLite debugger: per-event count of weakly-connected components and how many SimTrack/SimVertex are disconnected from a generator primary (the orphans). Exits non-zero if any event has orphans. `--link {parentIndex,ancestor,combined}`. |
| `../python/truthGraphSelections.py` | Per-process selection presets: maps a generator fragment (or label) to one of seven archetypes (gun / resonance / vbf / ggf / top / heavyflavor / full) and returns the right `postProcessing` selection. `selectionForFragment(name, **overrides)` (dict), `postProcessingPSet(...)` (cms.PSet), `dumperArgs(...)` / CLI (`python3 truthGraphSelections.py <fragment>`) emit the dumper flags. |
| `makeTruthGallery.sh` | Build the per-process DOT/SVG gallery (full + natural-seed selection) from a relval library dir; the per-sample selection is resolved by `truthGraphSelections.py` from each workflow's fragment. |
| `makeBranchValidationPlots.sh` | Render the Branch DQM validation plots (overlaying a few samples) from the per-workflow harvested DQM, via `scripts/makeTruthGraphValidationPlots.py`. |
| `runTruthRelvals.sh` | Run the 8 enableTruth Run4 D120 no-PU truth-validation workflows via `runTheMatrix`. |
| `TruthLogicalGraphPostProcessor_t.cpp` | cppunit tests for the logical-graph postprocessing (selection, merging, collapsing). |

## Typical flow
```bash
cmsenv                                   # from CMSSW_17_0_0_pre2/src
runTruthRelvals.sh  /path/library        # produce the sample library (step1..5)
truthGraphConnectivity.py /path/library/34050.88_*/step3.root   # sanity: orphans == 0
makeTruthGallery.sh /path/library /path/dot_gallery             # DOT + SVG gallery
makeBranchValidationPlots.sh /path/library /path/branch_plots   # Branch DQM validation plots
```

## Focused selections (phases 1-3)
The postprocessing supports focused, physics-oriented views:
- `--no-keepSpectators` drops underlying-event spectators, leaving the selection
  plus its truncated upstream attached to a labeled **ISR/upstream** source node.
  Spectators (when kept) sit on a separate **underlying event** node; both
  artificial nodes carry the genEvent/eventId of the activity they summarize
  (pile-up provenance).
- `-f/--flavors` seeds on hadrons by heavy-flavor content (`-f 5` = B hadrons,
  `-f 4` = D hadrons), OR-ed with `-s/--seeds`.
- `--keepProductionSiblings` keeps the seed's **hard-scatter co-products**: the
  other outgoing particles of its production vertex (and their subtrees). These are
  siblings of the seed, not ancestors, so `-d/--parentDepth` never reaches them -
  e.g. seeding on the Higgs in VBF, this brings in the recoiling tagging quarks and
  their forward jets, and shows the real hard vertex in place of the artificial
  Upstream node.
```bash
# clean Z -> mu mu view with an explicit ISR node:
cmsRun dumpTruthGraphsFromGENSIMRECO_cfg.py file:step3.root -s 23 -d 1 --no-keepSpectators
# all B-hadron decay subgraphs:
cmsRun dumpTruthGraphsFromGENSIMRECO_cfg.py file:step3.root -f 5 --no-keepSpectators
# VBF Higgs with the tagging quarks/jets that produced it:
cmsRun dumpTruthGraphsFromGENSIMRECO_cfg.py file:step3.root -s 25 -d 1 --keepProductionSiblings
```
Rather than remember the right flags per process, let `truthGraphSelections.py`
pick them from the generator fragment (seven presets, fully overridable):
```bash
# the dumper flags for any fragment / label:
python3 ../python/truthGraphSelections.py VBFHZZ4Nu_14TeV
#  -> -s 25 -d 1 --keepSpectators --attachSources --keepProductionSiblings
cmsRun dumpTruthGraphsFromGENSIMRECO_cfg.py file:step3.root \
       $(python3 ../python/truthGraphSelections.py ZMM_14)
```

## Jet -> originating particle
The graph answers "which particle did this jet come from" once you have the
jet's truth constituents (GEN-jet constituents now; hit-matched reco
constituents via the LogicalGraphHitIndex later):
```cpp
// jetParticles: the truth::Particle of each constituent
auto origin = graph.lowestCommonAncestor(jetParticles);   // e.g. the b quark of a b-jet
auto top    = jetParticles.front().firstAncestorWithPdgId(6);  // the originating top
```
`lowestCommonAncestor` returns the closest shared ancestor; `firstAncestorWithPdgId`
walks up to a specific origin species.

## One-off selection / coherence scan
`dumpTruthGraphsFromGENSIMRECO_cfg.py` drives all selection studies, e.g.:
```bash
# Z -> mu mu only (drop Z -> ee), depth-0 context:
cmsRun dumpTruthGraphsFromGENSIMRECO_cfg.py file:step3.root -s 23 -g 13,-13 -d 0
# full graph for debugging:
cmsRun dumpTruthGraphsFromGENSIMRECO_cfg.py file:step3.root -s 0 --showAll
```
