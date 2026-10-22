# HGCal Common-Mode DNN Correction as a RecHit Energy Variation (alpaka)

The common-mode (CM) noise correction from a PyTorch DNN runs inside the RecHit step
(`HGCalRecHitsProducer`, alpaka, CPU or GPU). It does **not** change the nominal reconstruction:
`energy` keeps the linear-regression CM correction, and the DNN result is stored next to it as an
alternative, `energy_variations(CM_ALT)`. Layer clusters get the same variation by re-summing
their member rechits (no re-clustering). NanoAOD users can pick either method per hit or cluster.

C++ lives in `RecoLocalCalo/HGCalRecAlgos` (RecHits, DNN) and `RecoLocalCalo/HGCalRecProducers`
(layer clusters, checked out locally); Python wiring, models, DQM and NanoAOD in `HGCalCommissioning/`.

History: until 2026-09 the DNN ran at the digi stage (`HGCalCMCalibrationProducer`, collection
`hgcalCMCalibDigis`) and subtracted its prediction from the integer digi ADC. That producer, its
`applyCMCorrections` kernel and the `HGCalCMCorrectionCompare` DQM were removed; see git history.

---

## Quick start

```bash
cd $CMSSW_BASE/src && cmsenv
cmsRun HGCalCommissioning/Configuration/test/step_RECONANODQM.py \
  era=TB2025/v6 run=112049 files=file:<RAW2DIGI file> maxEvents=-1 skipMeta=True \
  calibfile=HGCalCommissioning/Calibrations/TB2025/calib/level0_calib_params_v5.json \
  elossfile=HGCalCommissioning/Calibrations/TB2025/calib/hgcal_energyloss_setup2_v1.json \
  cmModel=zscore skipLC=False trigTimeMin=107 trigTimeMax=113 \
  output=NANO.root
```

- `cmModel`: `plain` (default) or `zscore`; `useCMML=False` turns the DNN off (`CM_ALT = energy`).
- `skipLC=False`: run layer clustering and write the `HGCLayerClusters` table with its variations.
- `trigTimeMin/Max`: optional event selection; 107–113 is the model's training selection.

For validation plots add `addCMComparison=True knoise=-100 cmCompareModules=ML-F3WC-IH0182`, or
run both models with `run_cm_comparison.sh` (see [Validation](#validation)). 2025 files need
`skipMeta=True`.

Reading the variation in NanoAOD:

```python
import uproot
t = uproot.open("NANO.root")["Events"]
e      = t["HGCHit_energy"].array()          # linear CM correction (nominal)
e_dnn  = t["HGCHit_energy_CM_ALT"].array()   # DNN CM correction
lc_dnn = t["HGCLayerClusters_energy_CM_ALT"].array()
```

---

## Per-event workflow

```
hgcalDigis (HGCalDigiHost)
   |  HGCalRecHitsProducer::acquire  (hgcalRecHits)
   |    copy digis + calibration to device
   |    [useCMML] fillModuleInputs   one thread per module: features 0..250
   |    [useCMML] fillDigiInputs     one thread per digi: copy module row, mask own ADC, features 251..253
   |    [useCMML] DNN inference      TorchScript via AlpakaModel, [ndigis, 254] -> [ndigis, 1]
   |    [useCMML, zscore model] denormalizeOutputs   per-channel output de-normalisation
   |    calibrate: flag, adcToEnergy (energy + energy_variations), toaToTime, calib cells, metadata
   |  HGCalRecHitsProducer::produce
   |    zero-suppression selection
hgcalRecHits (energy = linear CM, energy_variations(CM_ALT) = DNN CM)
   -> hgcalRecHitsLayerClusters (CLUE on nominal energy) -> hgcalSoALayerClusters (energy, time + variations)
   -> NanoAOD HGCHit / HGCLayerClusters
```

## Energies

For an ADC-mode hit, with `s = EM_scale * MIPS_scale` and the pedestal subtraction shared:

```
pedsub    = adc   - ADC_ped
pedsub_m1 = adcm1 - ADC_ped
cm_lin    = CM_slope * (0.5*cm - CM_ped)

energy                         = s * ((pedsub - cm_lin) - BXm1_slope*(pedsub_m1 - cm_lin))
energy_variations(CM_ALT)      = s * (pedsub - NN)          # no BX-1 term
energy_variations(SCALE_ALT)   = energy                     # not defined yet
mipEnergy                      = MIPS_scale * (nominal counts)
```

- `NN` is the DNN prediction in pedestal-subtracted ADC (subtractive, confirmed with the model
  author Arne Reimers). It is a float: nothing is rounded into the digi ADC any more.
- `CM_ALT = energy` where the DNN does not apply: TOT-mode hits, invalid digis, unconnected cells
  (cell-area SF 0), modules outside the model scope (not 222 channels), or `useCMML=False`.
- `BXm1_slope` is 0 in every TB2025 calibration.
- Downstream of the calibration (ZS selection, timewalk, CLUE, LC energy/position) only the
  nominal `energy` is used. There is no `mipEnergy` variation: `nmips_alt = nmips * energy_alt / energy`.
- Calibration cells merge their energy into a partner cell; every variation is merged the same way.
- `cmAltAsNominal=True` (producer parameter, requires `useCMML=True`, not exposed in the python
  options) swaps the two: `energy` gets the DNN CM, so ZS and clustering run on it, and
  `energy_variations(CM_ALT)` gets the linear CM. Off by default. `HGCalCMVariationValidation`
  assumes the default and mislabels its spectra with the swap on.

Variation indices (`DataFormats/HGCalReco/interface/HGCalRecoVariations.h`):

```cpp
namespace hgcalreco::Variation { constexpr uint8_t CM_ALT = 0x0, SCALE_ALT = 0x1; constexpr uint8_t Size = 0x2; }
using VariationVector = Eigen::Matrix<float, Variation::Size, 1>;
```

Values are **absolute** alternative energies/times, not deltas.

## Data formats

| SoA | New columns |
|---|---|
| `HGCalSoARecHits` | `SOA_EIGEN_COLUMN(VariationVector, energy_variations)` |
| `HGCalSoAClusters` | `time`, `SOA_EIGEN_COLUMN(VariationVector, energy_variations)`, `SOA_EIGEN_COLUMN(VariationVector, time_variations)` |
| `HGCalSoAClustersExtra` (workspace) | `time_weight`, `time_weighted_sum` and their `_variations` |

SoA headers with Eigen columns need `<Eigen/Core>` included before any SoA header; the HGCalReco
SoA headers now include it themselves.

`RecoLocalCalo/HGCalRecProducers` is checked out because the layer-cluster kernels live there
and because the release build of it uses the old RecHit SoA column offsets.

## Layer clusters

In `HGCalLayerClustersSoAAlgoWrapper.dev.cc`, over rechits with `clusterIndex != -1`:

```
energy_variations(k) = sum of member rechit energy_variations(k)
time                 = sum(w*t) / sum(w),  w = max(E, 0),        rechits with time > 0;  -99 if sum(w) == 0
time_variations(k)   = sum(w_k*t) / sum(w_k), w_k = max(E_k, 0)  (same rechit times, no timewalk redo)
```

The clusters themselves (membership, position) come from the nominal energy.

With a negative zero-suppression threshold (`knoise=-100`) a cluster can have no member with
energy > 0. The release position kernels then read rechit index −1 and the job aborts (run 112049,
event 1553). The position kernels now skip such clusters and set x = y = z = 0 (position undefined;
about 1 in 6000 clusters on run 112049); energy, variations and time are unaffected.

With a positive threshold (e.g. `knoise=3`) an event can have no rechits at all. CLUE never writes
`numberOfClustersScalar` then, and the uninitialised value sized the cluster collection (crash or
segfault at run 112049 event 116490). `HGCalSoARecHitsLayerClustersProducer` now zeroes the count
and skips CLUE for empty input.

Rechits are selected on the nominal energy, so with `knoise > 0` a hit's `energy_CM_ALT` (and hence
a cluster's) can still be below the threshold or negative. The legacy
`timeLayerCluster` value map from `HGCalLayerClustersFromSoAProducer` is untouched (on TB data
it is not usable: rechit `timeError` is 0).

## DNN inputs (254 floats per digi)

Model manifest `feature_version: all_channels_multimodule_v1`, `nch: 222`, trained on the ten
TB2025 LD modules. Layout in `HGCALSoACMML.h` (named offsets in `hgcalcmml::Feature`), one Eigen
column so `TensorCollection` wraps it into `[ndigis, 254]` without copying.

| Index | Feature | Definition |
|---|---|---|
| 0–11 | `cm_erx00..11_pedsub` | `2*(0.5*cm - CM_ped)` from the first channel of each eRx; 0 if eRx disabled or `valid==0` |
| 12 | `nchtoa` | module channels with `flags != NotAvailable && toa > 0` |
| 13 | `nchtot` | module channels with `flags != NotAvailable && tctp == 3 && tot > 0` |
| 14–17 | `nchadcgt10/50/200/500` | module channels with pedsub ADC `> X`, all 222 incl. unconnected and the target |
| 18–28 | `module_onehot_000..010` | index of the typecode in the manifest `module_vocabulary` (`-` → `_`); all 0 for unknown modules (slot 10 `UNKNOWN` never set) |
| 29–250 | `adc_allch_000..221` | `adc - ADC_ped` of every module channel, 0 if `valid==0`; **the target channel's own slot is 0** |
| 251 | `channel_indices` | `chIdx - (nErx*37 - 1)/2` (±110.5) |
| 252 | `erx_indices` | `erxIdx - (nErx - 1)/2` |
| 253 | `cell_area_fraction` | SF from `cellareas.json` (`ML_F`/`MH_F` by `isHD`), 0 = unconnected |

Features 0–250 are identical for every digi of a module, so they are built once per module and
copied. The producer throws at construction if the manifest feature names or order differ from
what the kernels fill.

Pedestals: all inputs use `ADC_ped`/`CM_ped` from the calibration JSON loaded for the job, the
same as the nominal energy (the model was trained with Run112044 pedestals; not checked against v5).

**Calibration validity guard.** Channels with `valid == 0` have no fitted pedestal (`ADC_ped` and
`CM_ped` are 0). Their CM and ADC features are 0 rather than the raw value. Without this, module
`ML-F3WC-IH0199`'s dead eRx 5 once put `cm5 ≈ +529` into every channel of that module.

## Models

| `cmModel` | Files in `RecoLocalCalo/HGCalRecProducers/data/CMCalibrations/` | Source (Arne Reimers, `dnn_best.pth`) |
|---|---|---|
| `plain` (default) | `cm_dnn_multimodule.pth`, `_manifest.json` | `allchannels_multimodule_perchannel_F254__256-256-256-32__dr0`, md5 `823ecb38…` |
| `zscore` | `cm_dnn_multimodule_zscore.pth`, `_manifest.json`, `_preprocessing.json` | `allchannels_multimodule_inputzscore_perchannel_F254__256-256-256-32__dr0`, md5 `b5116811…` |

Source directory: `/eos/user/a/areimers/hgcal/Sep2025TB/corrections/MULTI_ML_F3WC_IH0180_…_IH0199/pedestals_from_Run112044/corrections_from_ModuleMULTI_…/corrections_from_Run112044_112050_112060_112073_adcmax10/corrections_tag_selection_trigtime/dnn_models/`.

- Architecture `Linear(254,256)-ReLU-Linear(256,256)-ReLU-Linear(256,256)-ReLU-Linear(256,32)-ReLU-Linear(32,1)`.
- Arne's checkpoints are `state_dict`s; `NNBased/convert_multimodule_dnn.py` rebuilds the network,
  checks the md5, scripts it and checks TorchScript == eager (max |diff| 0). Run under `cmsenv`.
- Normalisation is applied in CMSSW, not baked into the model file.
- The model files and `cellareas.json` are shipped in the cms-data package
  [`RecoLocalCalo-HGCalRecProducers`](https://github.com/cms-data/RecoLocalCalo-HGCalRecProducers)
  under `CMCalibrations/`. The `hgcalRecHits` defaults point to the `plain` model there. To test
  a new version before it is in a release, clone the package into
  `$CMSSW_BASE/external/$SCRAM_ARCH/data/RecoLocalCalo/HGCalRecProducers/data`.

**z-score model.** From `input_preprocessing.json`:

```
x'  = (x - mean[i]) / std[i]                        all 254 inputs, one-hots included
own adc_allch slot: raw 0, then normalised          (-mean/std)
NN  = pred * target_std[chIdx] + target_mean[chIdx] per-channel output de-normalisation
```

Both conventions confirmed 2026-09-16 and hardcoded. The producer requires a
preprocessing file when the manifest says `preprocess_inputs: true` and rejects one otherwise.

## Configuration

`HGCalCommissioning/Configuration/python/configure_sysval_reco_cff.py` builds `hgcalRecHits`
(cuda and serial) on `hgcalDigis`. Options (`setArgParserForRECO`):

| Option | Default | Meaning |
|---|---|---|
| `useCMML` | `True` | run the DNN and fill `CM_ALT`; `False`: `CM_ALT = energy`, no model loaded |
| `cmModel` | `plain` | `plain` or `zscore` (table above) |

`step_RECONANODQM.py` adds:

| Option | Meaning |
|---|---|
| `addCMComparison` | schedule `HGCalCMVariationValidation` + harvester (needs `useCMML=True`; use `knoise=-100`) |
| `cmCompareModules` | typecodes with their own per-module spectra |
| `debugCsv`, `debugEvent/Module/MaxChannels/FedId/CaptureBlock/Econd` | DNN input dump (below) |
| `trigTimeMin/Max`, `trigTimeRequireValid` | trig-time event selection, unchanged (model trained on 107 < t < 113) |

`hgcalRecHits` parameters: `useCMML`, `cmAltAsNominal`, `model`, `modelManifest`, `inputPreprocessing`,
`cellAreas`, `cm_n_threads`,
`moduleIndexerSource`, `produceCMMLDebug`. Paths are CMSSW-relative (resolved with `FileInPath`).

## NanoAOD

`HGCalCommissioning/NanoTools/plugins/HGCalRecoTableProducer.cc`:

- `HGCHit_energy_CM_ALT`, `HGCHit_energy_SCALE_ALT`
- `HGCLayerClusters_energy_CM_ALT`, `_energy_SCALE_ALT`, `_time`, `_time_CM_ALT`, `_time_SCALE_ALT`

A FlatTable column is scalar, so each variation entry is its own branch.

The `HGCLayerClusters` table is written only with `skipLC=False`. (Before 2026-09-16
`configure_sysval_nano_cff.py` never passed `skipLC` to `hgCalRecoTable`, so the table was never
written; fixed.) The pre-existing per-rechit columns in that table (`rho`, `delta`,
`clusterIndex`, …) and `seed` (filled from `cells`) are still unreliable.

## Validation

The DQM module and `run_cm_comparison.sh` are in `HGCalCommissioning`. The Python analysis
scripts named below (`plot_adc_shape_from_dqm.py`, `verify_smoke.py`, `verify_lc.py`,
`reference_population_check.py`) and the output folders are **local to the development area and
not in either repository**; the results are kept here as a record of the checks.

**DQM:** `HGCalCMVariationValidation` (+ `HGCalCMVariationValidationHarvester`, folder
`HGCal/CMComparison`) reads the rechits back:

```
counts_lin  = energy / s
counts_nn   = energy_variations(CM_ALT) / s
counts_noCM = counts_lin + cm_lin*(1 - BXm1_slope)
dnn_corr    = (adc - ADC_ped) - counts_nn
```

It skips TOT hits and calibration cells with their partners, and books spectra over all rechits
and connected cells only (`*Conn`): `adcPedSubRaw/Analytic/MLFloat`, the same three in GeV
(`energyNoCM/Linear/CMALT`), `analyticCorrection`, `mlCorrectionFloat`, `residual`,
`analytic_vs_mlfloat` (Pearson r in the title), `ml_vs_cmsum`, `energyCMALTminusNominal`,
`rechitCounts` (seen / TOT / calib-related / used / connected). `corrStats` and `rechitCounts`
are double-precision (a full run exceeds the 2^24 exact range of float bins). Only existing
rechits are seen, so run with `knoise=-100`.

`HGCalCommissioning/Configuration/test/run_cm_comparison.sh` runs one job per model into
`<outdir>/<model>/`:

```bash
bash src/HGCalCommissioning/Configuration/test/run_cm_comparison.sh \
  --era TB2025/v6 --run 112049 --files src/HGCalCommissioning/Configuration/test/fileList_112049.txt \
  --maxevents -1 --skipmeta --trigtimemin 107 --trigtimemax 113 --nobuild \
  --models plain,zscore --modules ML-F3WC-IH0182 --outdir <outdir>
```

Fit the spectra with `plot_adc_shape_from_dqm.py` (local), which writes one PNG per
correction (`adcshape_no_cm_correction.png`, `adcshape_lin_reg_correction.png`,
`adcshape_dnn_correction.png`) with the Gaussian fit over [-10, 10] ADC:

```bash
python3 plot_adc_shape_from_dqm.py --connected --module ML-F3WC-IH0182 \
  --dqm <outdir>/<model>/DQM_V0001_HGCAL_R000112049.root --outdir <outdir>/<model>/adcshape_IH0182_conn
```

Omit `--module` for all modules pooled.

**Cross-check script:** `verify_smoke.py <job dir> <plain|zscore>` (local) takes a job's debug CSV and NANO and checks the input definitions, re-runs the model in PyTorch
on the same rows, and checks `energy_SCALE_ALT == energy`, TOT hits, and
`(adc - ADC_ped) - CM_ALT counts == DNN correction`. Passed for both models on 20 events of run
112049 (2026-09-16; max |diff| ~1e-4 ADC).

Nominal `energy` agrees with the pre-change code on the same events to float32 rounding (typical
relative difference 1e-7): the restructured pedestal/CM expression rounds differently in the last bit.

`verify_lc.py <NANO>` (local; job with `skipLC=False`) recomputes every cluster's
energy, `energy_CM_ALT`, `time` and `time_CM_ALT` from the member HGCHit rows; passed on 50
events (worst relative difference 5e-4, float32 sums in parallel).

**Full run 112049** (2026-09-16, 107 < trig_time < 113, `knoise=-100`, v5 calib, fit over
[-10, 10] ADC, connected cells):

| Population | no CM μ / σ | linear μ / σ | DNN `plain` μ / σ | DNN `zscore` μ / σ |
|---|---|---|---|---|
| ML-F3WC-IH0182 | −0.465 / 1.932 | 0.832 / 2.323 | −0.181 / 1.416 | −0.259 / 1.265 |
| all modules | −0.420 / 1.723 | 0.533 / 1.745 | −0.121 / 1.310 | −0.204 / 1.185 |

The earlier digi-based reference for IH0182 (no CM 1.879, linear 2.249) was over a different
population: it also contained calibration cells and their partners, which the rechit-based
analyzer skips. `reference_population_check.py <NANO>` (local) rebuilds both
populations from the new job's digis and gives exactly 1.879 / 2.249 (with) and 1.932 / 2.323
(without), so the nominal reconstruction is unchanged on the full run.

With `knoise=-100` CLUE also clusters noise, so the largest clusters span most of a module
(150–190 cells) and their `energy_CM_ALT - energy` reaches a few GeV from the DNN's mean offset
summed coherently; clusters below 50 cells have median |ΔE| 0.018 GeV. LC times inherit the
rechit times, which are unusable while the timewalk parameters are wrong (below).

## Debugging DNN inputs

`debugCsv=<file>` (`step_RECONANODQM.py`) sets `hgcalRecHits.produceCMMLDebug=True`, which puts four
host products in the event (instance labels on `hgcalRecHits`; empty when the DNN did not run):

| Label | Type | Content |
|---|---|---|
| `cmmlFeatures` | `std::vector<float>` | ndigis × 254 inputs as fed to the model (normalised for `zscore`), row-major |
| `cmmlOutput` | `std::vector<float>` | raw model output per digi |
| `cmmlCorrection` | `std::vector<float>` | correction in ADC (after de-normalisation) |
| `cmmlValid` | `std::vector<unsigned char>` | whether the correction is applied |

`HGCalCMMLDebugDumper` (`HGCalCommissioning/DQM`) writes them to the CSV, one row per (event, module,
channel) passing `debugEvent` (0 or 4294967295 = all) / `debugModule` / hardware address /
`debugMaxChannels`:

```
event,denseModIdx,module,ch,digi_adc,<254 feature names from the manifest>,dnn_output,dnn_corr,dnn_valid
```

Features are **raw** (normalisation undone, masked slot 0) for both models; `dnn_output` is the
model output, `dnn_corr` the correction in ADC, `dnn_valid` whether it is applied. The dumper is on a
`Path`, so the trig-time selection applies to it. For IH0182 on run 112049 the dense module index is 6.

## Indexing

- Dense index = global channel = digi index; `HGCalDenseIndexInfo` gives `modInfoIdx`,
  `chNumber` (within module) and `cellInfoIdx` per entry. TB2025: 10 modules × 222 = 2220.
- `chDataOffsets_[modid]` is the first dense index of a module. Size module arrays with
  `maxModulesCount()` (10), not `maxModuleSize()` (distinct typecodes, 1).
- `enabledErx_` is a bitmask; eRx count = `__builtin_popcount`.
- Rechits are compacted by the ZS selection; `recHitIndex` is the dense index.

## Files

Paths relative to `src/`.

**New**
- `DataFormats/HGCalReco/interface/HGCalRecoVariations.h` — `Variation::CM_ALT/SCALE_ALT`, `VariationVector`
- `HGCalCommissioning/DQM/plugins/HGCalCMVariationValidation.cc`, `HGCalCMVariationValidationHarvester.cc`
  (+ `DQM/python/hgcalCMVariationValidation[Harvester]_cfi.py`)
- `HGCalCommissioning/Calibrations/NNBased/convert_multimodule_dnn.py`
- cms-data `RecoLocalCalo/HGCalRecProducers/data/CMCalibrations/`: `cm_dnn_multimodule[_zscore].pth`,
  `…_manifest.json`, `cm_dnn_multimodule_zscore_preprocessing.json`, `cellareas.json`
- `HGCalCommissioning/SystemTestEventFilters/plugins/HGCalTrigTimeFilter.cc` — trig-time selection

**Modified**
- `DataFormats/HGCalReco/interface/HGCalSoARecHits.h`, `HGCalSoAClusters.h` — variation/time columns;
  `HGCalSoARecHitsExtra.h`, `MtdSoA.h` — Eigen pre-include
- `RecoLocalCalo/HGCalRecAlgos/interface/HGCALSoACMML.h` — 254-float Eigen column + `Feature` offsets
- `RecoLocalCalo/HGCalRecAlgos/interface/HGCalCMCorrectionSoA.h` — adds `valid`
- `RecoLocalCalo/HGCalRecAlgos/{interface,plugins}/alpaka/HGCalCMCalibrationAlgorithms.*` — module/digi input kernels, output de-normalisation
- `RecoLocalCalo/HGCalRecAlgos/{interface,plugins}/alpaka/HGCalRecHitCalibrationAlgorithms.*` — pedestal/linear/DNN split, variations, calib-cell merge
- `RecoLocalCalo/HGCalRecAlgos/plugins/alpaka/HGCalRecHitProducers.cc` — DNN inputs, inference, manifest/normalisation, optional debug products
- `RecoLocalCalo/HGCalRecProducers/interface/HGCalSoAClustersExtra.h`,
  `plugins/alpaka/HGCalLayerClustersSoAAlgoWrapper.dev.cc`, `plugins/alpaka/HGCalSoARecHitsProducer.cc` — LC variations and time
- `HGCalCommissioning/NanoTools/plugins/HGCalRecoTableProducer.cc`, `NanoTools/python/configure_sysval_nano_cff.py`
- `HGCalCommissioning/Configuration/python/configure_sysval_reco_cff.py`, `Configuration/test/step_RECONANODQM.py`,
  `Configuration/test/run_cm_comparison.sh`, `DQM/plugins/BuildFile.xml`
- `HGCalCommissioning/DQM/plugins/HGCalCMMLDebugDumper.cc` — debug CSV from the `produceCMMLDebug` products

**Removed**
- `RecoLocalCalo/HGCalRecAlgos/plugins/alpaka/HGCalCMCalibrationProducer.cc`
- `HGCalCommissioning/DQM/plugins/HGCalCMCorrectionCompare[Harvester].cc` and their cfis

## Known limitations

- Only 222-channel (6 eRx, LD) modules get a DNN prediction.
- The model was trained on trig-time-selected events (107 < t < 113); reco applies it to all events.
- `moduleIndexer.enabledErx_` is the module type's full mask, not the run configuration
  (`HGCalConfiguration`); partial masks (P5, FNAL) would mis-centre the indices and CM features.
- Module-level host arrays are rebuilt and uploaded every event.
- Timewalk parameters (`TOA_TW`, e.g. `[286.4, 25968.8, 0]`) look inconsistent with GeV input to
  `log(E)`: rechit and LC times come out at O(1e6) ps on run 112049. Not addressed here. LC time
  variations do not redo timewalk; LC x/y are not varied.
- The CUDA backend compiles but has not been run (no GPU on the lxplus node used).
- Namespace-scope `constexpr` arrays are not usable in CUDA kernels (use a local copy).
- CMSSW torch has no NumPy interop: use `torch.tensor(list)` / `.tolist()`.
