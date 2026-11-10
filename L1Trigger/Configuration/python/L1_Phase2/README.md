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
| `psets/`     | Top-level PSets referenced by modules with `refToPSet_` (P2GT cos/cosh LUTs, HGCal noise and charge collection efficiencies) |
| `services/`  | empty: no service is needed |

The same content is also written to a single file, `../L1_Phase2_expanded_cff.py`, with every
object defined inline (like the Run-3 HLT menu dumps, e.g. `HLT_GRun_cff.py`): convenient to read
the whole menu or diff two versions. Load one or the other, never both (same labels).

`L1Trigger/Configuration/test/OnLine_L1_Phase2.py` is a complete process (source, GlobalTag, geometry,
output) running only the L1 emulation, i.e. the cmsDriver step `L1P2` with the customisation
`customisePhase2TTNoMC.customiseL1P2NoMCTruth` (see below), on the GEN-SIM-DIGI-RAW output of the
reference workflow, like `HLTrigger/Configuration/test/OnLine_HLT_GRun.py` for the HLT:

    cmsRun OnLine_L1_Phase2.py   # reads file:step2.root, writes file:OnLine_L1_Phase2.root

## Generated, do not edit by hand

All files are produced by `L1Trigger/Configuration/test/l1tPhase2Fragment.py`. Changes belong in
the live `_cfi`/`_cff` files, followed by a regeneration:

    l1tPhase2Fragment.py generate   # rewrite L1_Phase2_cff.py, L1_Phase2/, L1_Phase2_expanded_cff.py, test/OnLine_L1_Phase2.py
    l1tPhase2Fragment.py validate   # compare all three with cmsDriver (unit test testL1Phase2Fragment)

For `OnLine_L1_Phase2.py` only the L1 content (without the MC-truth path) is compared, not the GlobalTag
or the I/O, so an update of the autoCond GlobalTag alone does not fail the check.

The reference is workflow 38434.0 (TTbar, Run4D128, era `Phase2C26I13M9`): the content is the
difference between `cmsDriver -s DIGI:pdigi_valid` and `cmsDriver -s DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT`.

* Only objects reachable from the L1 paths are kept; modules, PSets and VPSets that the cffs
  load but nothing schedules or references are dropped.
* Geometry, magnetic field and GlobalTag are not included; they come from the driver.
* L1-specific ES sources (including `l1ugmtdb`) are included in `eventsetup/`.
* The Run-3 legacy chain (everything that only feeds `simGtStage2Digis`, `simBmtfDigis` or
  `simCscTriggerPrimitiveDigisRun3`) runs in its own `L1TLegacyRun3Task` / `L1simulationLegacy_step`.
* Module labels are identical to the cmsDriver ones (DQM, ntuplizers, HLT inputs).
* The MC-truth chain (the `TT*Associator*` modules and everything reading their products, currently
  also `DisplacedVertexProducer`) runs in its own `L1TrackTriggerMCTruthTask` / `L1TrackTriggerMCTruth_step`.
  Nothing else may depend on it; the generator fails otherwise. Inputs that a module reads only when a
  flag is set (`L1FPGATrackProducer` with `readMoreMcTruth`) are not counted as dependencies.
* The required external inputs are listed in the header of `L1_Phase2_cff.py`.

## Validity

Generated for the default Run4D128 (`Phase2C26I13M9`) and identical for all geometries of that era
(D120, D122, D123, D126, D127, D128). It does not match `Phase2C17I13M9`, `Phase2C22I13M9` or
`Phase2C20I13M9` (HGCal TPG configuration, HFNose TPG modules for C20).

## Usage

With cmsDriver, the step `L1P2` replaces `L1TrackTrigger,L1,L1P2GT` (`L1P2:<name>` loads
`L1Trigger/Configuration/L1_<name>_cff`, default `Phase2`):

    cmsDriver.py step2 -s DIGI:pdigi_valid,L1P2,DIGI2RAW,HLT:@relvalRun4 ...

`L1P2:Phase2_expanded` loads the single-file version instead; the resulting configuration is identical.

To run the L1 emulation alone on stored events (e.g. GEN-SIM-DIGI-RAW), add
`--customise L1Trigger/Configuration/customisePhase2TTNoMC.customiseL1P2NoMCTruth`: it drops the MC-truth
path (the tracker sim links read by the associators are not stored) and loads the HCAL trigger primitive
configuration, whose ES producers the L1 emulation needs and the DIGI step otherwise provides.

    cmsDriver.py l1 -s L1P2 --customise L1Trigger/Configuration/customisePhase2TTNoMC.customiseL1P2NoMCTruth \
      --filein file:step2.root --processName L1P2 ...

The matrix workflows `<N>.783` of the `Phase2C26I13M9` geometries (e.g. `38434.783`, Run4D128) use it.

In a configuration, load the fragment and add its paths (listed in schedule order in `l1Paths`);
loading the fragment does not change `process.schedule`:

    from L1Trigger.Configuration.L1_Phase2_cff import l1Paths
    process.load("L1Trigger.Configuration.L1_Phase2_cff")
    for i, label in enumerate(l1Paths):
        process.schedule.insert(index + i, getattr(process, label))  # e.g. after the digitisation step
