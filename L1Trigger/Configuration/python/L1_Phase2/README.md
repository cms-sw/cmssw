# L1_Phase2: standalone Phase-2 L1 emulation configuration

Fully resolved configuration of the Phase-2 L1 emulation (track trigger, trigger primitives,
correlator/GTT/GMT/calo emulators and the P2GT menu) loaded by `../L1_Phase2_cff.py`
(a `cms.ProcessFragment("L1")`), analogous to `HLTrigger/Configuration/python/HLT_75e33/`.
Tracking issue: https://github.com/artlbv/cmssw/issues/10

| Directory    | Content |
|--------------|---------|
| `modules/`   | EDProducers/EDFilters, one `<label>_cfi.py` each |
| `eventsetup/`| ES sources, ES producers and ES prefers |
| `sequences/` | Sequences and Tasks |
| `paths/`     | Paths (track trigger, L1 emulation, Run-3 legacy, GT producers, one `p<X>` path per GT algorithm) |
| `psets/`, `services/` | empty: no shared PSet or service is needed |

## Generated, do not edit by hand

All files are produced by `L1Trigger/Configuration/test/l1tPhase2Fragment.py`. Changes belong in
the live `_cfi`/`_cff` files, followed by a regeneration:

    l1tPhase2Fragment.py generate   # rewrite L1_Phase2_cff.py and L1_Phase2/
    l1tPhase2Fragment.py validate   # compare with cmsDriver (unit test testL1Phase2Fragment)

The reference is workflow 29634.0 (TTbar, Run4D110, era `Phase2C17I13M9`): the content is the
difference between `cmsDriver -s DIGI:pdigi_valid` and `cmsDriver -s DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT`.

* Only objects reachable from the L1 paths are kept; modules, PSets and VPSets that the cffs
  load but nothing schedules or references are dropped.
* Geometry, magnetic field and GlobalTag are not included; they come from the driver.
* L1-specific ES sources (including `l1ugmtdb`) are included in `eventsetup/`.
* The Run-3 legacy chain (everything that only feeds `simGtStage2Digis`, `simBmtfDigis` or
  `simCscTriggerPrimitiveDigisRun3`) runs in its own `L1TLegacyRun3Task` / `L1simulationLegacy_step`.
* Module labels are identical to the cmsDriver ones (DQM, ntuplizers, HLT inputs).
* MC-truth modules (`TT*Associator*`) are still part of `L1TrackTrigger_step`; the emulators
  themselves do not read truth with the default settings.
* The required external inputs are listed in the header of `L1_Phase2_cff.py`.

## Usage

Loading the fragment replaces `process.schedule` with the L1 schedule, as for `HLT_75e33_cff`:

    schedule = process.schedule
    process.load("L1Trigger.Configuration.L1_Phase2_cff")
    l1Paths = list(process.schedule)
    process.schedule = schedule
    # insert l1Paths after the digitisation step
