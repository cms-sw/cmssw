# Minimal OO/ppRef EEC forest content

This contract was rechecked against every file in
[energyEnergyCorrelatorAnalysis/src at 792d76c](https://github.com/JesusVlz/energyEnergyCorrelatorAnalysis/tree/792d76c0cd043ab1871b3217ac988f6455f2d9f0/src)
and against the uploaded OO data file `L1_object_data_6040.root`.

Use this producer revision with [the companion EEC reader update](https://github.com/JesusVlz/energyEnergyCorrelatorAnalysis/commit/b2d61d8840614fe8a966b1e53501131545ba1278) that makes
the five unused Run-3 legacy track fields optional. The older EEC main still
opens those fields unconditionally. The change preserves PbPb reader behavior.

All four production configurations keep R4/R8, `jetPtMin=15` GeV,
`jetAbsEtaMax=2.5`, WTA and the current PF Jet ID/veto inputs.
OO additionally retains the complete heavy-ion jet composition block and
the MiniAOD event-level scalar rho for possible future corrections. These are
retention settings, not proof of veto coverage outside that eta range.
All event trees remain in the same accepted-event order.

## Exact ROOT input contract

| ROOT tree | Sample | Retained branches |
| --- | --- | --- |
| `hiEvtAnalyzer/HiTree` | All | `run, lumi, evt, vz, hiHFPlus_pf, hiHFMinus_pf` |
| Same | OO additions | `hiBin, hiHF_pf` |
| Same | MC additions | `pthat, weight` |
| `hltanalysis/HltTree` | OO data | `HLT_MinimumBiasHF_OR_BptxAND_v1` |
| Same | ppRef data | `HLT_PPRefZeroBias_v6` |
| `skimanalysis/HltTree` | All | `pprimaryVertexFilter` |
| Same | OO data additions | `OOpfCoincFilterPF2Th4, pileupVertexFilter` |
| Same | OO MC additions | `OOphfCoincFilterPF2Th4, pileupVertexFilter` |
| `PbPbTracks/trackTree` (OO) or `ppTracks/trackTree` (ppRef) | All | `nTrk, trkPt, trkEta, trkPhi, trkPtError, highPurity, trkDzFirstVtx, trkDzErrFirstVtx, trkDxyFirstVtx, trkDxyErrFirstVtx, trkCharge` |
| `akCs4PFJetAnalyzer/t`, `akCs8PFJetAnalyzer/t` (OO) or `ak4PFJetAnalyzer/t`, `ak8PFJetAnalyzer/t` (ppRef) | All | `nref, rawpt, jtpt, jteta, jtphi, WTAeta, WTAphi, trackMax, jtarea, jtPfCHF, jtPfNHF, jtPfCEF, jtPfNEF, jtPfMUF, jtPfCHM, jtPfNHM, jtPfCEM, jtPfNEM, jtPfMUM` |
| Same jet trees | All PF jet additions | `jtPfCM, jtPfNM` |
| Same jet trees | OO additions | `trackSum, trackN, trackHardSum, trackHardN, chargedMax, chargedSum, chargedN, chargedHardSum, chargedHardN, photonMax, photonSum, photonN, photonHardSum, photonHardN, neutralMax, neutralSum, neutralN, eMax, eSum, eN, muMax, muSum, muN` |
| Same jet trees | MC additions | `refpt, refeta, refphi, refparton_flavorForB, ngen, genpt, geneta, genphi, WTAgeneta, WTAgenphi` |
| `HiGenParticleAna/hi` | MC | `pt, eta, phi, chg, sube` |
| `ggHiNtuplizer/EventTree` | OO and ppRef | Scalar `rho` from `fixedGridRhoFastjetAll`, with `run, event, lumis` for alignment/provenance |
| `HiForestInfo/HiForest` | All | Existing small metadata payload; its presence is also used by GeneratorLevelForestReader for format detection |

The track and generator-particle lists have 11 and 5 branches. Jet trees
have 44/54 branches for OO data/MC and 21/31 for ppRef data/MC respectively. Canonical physical event identity
stays in HiTree with unsigned 64-bit `evt`; duplicate/truncated jet IDs
are omitted.

The ordinary OO/ppRef cuts return before the PbPb chi2/hit/calo cuts.
Thus `trkNormChi2, trkNLayers, trkNHits, pfEcal, pfHcal` are no longer
stored. The updated HighForestReader reads them if present, tolerates their
absence only for OO/ppRef, and reports an error if an absent value is
explicitly requested. UnfoldingForestReader already treats them as optional.

Track retention remains pT >= 0.5 GeV and |eta| <= 2.6 over the full event,
with `applyTrackSelections=False`. Generator output keeps stable charged
particles with pT >= 0.5 GeV and the existing |eta| <= 2.5 range. Generator-jet
reconstruction has its own input, independently of the particle-output cut.
`sube` filling and its missing-product sentinel are preserved.

## Full PF multiplicities and OO jet composition

The original five category counts omit HF hadron and HF EM constituents.
Both OO and ppRef slim configurations save native Int_t[nref] branches `jtPfCM` and
`jtPfNM`, filled from `pat::Jet::chargedMultiplicity()` and
`pat::Jet::neutralMultiplicity()`. Their sum is the constituent count used by
CMSSW's [PFJetIDSelectionFunctor](https://github.com/JesusVlz/cmssw/blob/35f008fbb78ac9bca6db693047de4b9d64bd8126/PhysicsTools/SelectorUtils/interface/PFJetIDSelectionFunctor.h#L280).
No geometric or category-only total is inferred.

Use the matching EEC reader/Jet-ID update. It prefers the exact pair for
Tight/TightLeptonVeto and MiniPOGpp. The legacy HINpp prescription continues
to use its explicitly defined category sums. Old forests without the exact
pair retain the lower-bound guard: unresolved thresholds still raise an error.
The pair must be present together, with stable types/presence across a chain.
Previously produced forests need regeneration to resolve this missing input.
This adds two integers per jet for OO and ppRef and changes no Jet ID
threshold or JEC payload. OO also retains all 24 `doHiJetID` composition
branches (including `trackMax`, which the EEC reader already uses) for both
R4 and R8, even when the nominal OO ID does not read every field.

## Triggers and filters

Data cards use TriggerSelection 8. The legacy getter
`GetCaloJet15FilterBit()` carries the minimum-bias/zero-bias decision;
it does not impose a 15 GeV jet-trigger requirement. The exact saved
version must match the reader. OO upstream event filtering keeps
`HLT_MinimumBiasHF_OR_BptxAND_v*`; `HLT_OxyL1SingleJet*` stays commented out.

Current MC cards use TriggerSelection 0, so the MC forest paths no longer
schedule hltanalysis. Enabling a trigger requirement for MC would also
require restoring its producer and matching HLT branch.

Skim output saves three decisions for OO and one for ppRef. Extra threshold
bits, cluster compatibility, pp scraping, `forest` and `superFilterPath`
are not saved. The latter can still control upstream event selection:
branch booking selection does not change FilterAnalyzer's superFilters logic.
Current OO/ppRef readers set the unused noise/scraping/cluster bits to pass.

## FastJet and PF-HF inputs

The optional analysis flag is `Run3UsePpL1FastJet 1` with `Run3JEC 1`.
The implementation currently supports ppRef R4. The same input payload is
retained in OO for a possible future correction. The ppRef correction uses:

- `ggHiNtuplizer/EventTree/rho`: scalar Float_t/Double_t, copied from
  `fixedGridRhoFastjetAll`;
- `ak4PFJetAnalyzer/t/jtarea`: positive measured Float_t/Double_t array
  indexed by Int_t `nref`, filled from `jet.jetArea()`;
- `rawpt` and standard detector `jteta`, already retained.

The measured area is also retained for R8. Producing it does not extend the
current analysis calibration to R8 or OO. No zero-rho, geometric-area, or
branch-renaming fallback is introduced.

OO data and MC now schedule `ggHiNtuplizer` with object output disabled,
reading `fixedGridRhoFastjetAll` and writing the same four branches as
ppRef: `run, event, lumis, rho`. The eta-binned `rhoAnalysis/t` tree
and OO random-cone trees remain removed. Saving the scalar does not yet
enable an OO L1FastJet correction in the EEC analysis. pp centrality
remains off.
The required pp PF-HF sums are now populated with `doPFHF=True` independently
of centrality; these HF sums are not the L1FastJet rho input.

## Inspection of L1_object_data_6040.root

The supplied file has 26,708 events and 12 trees. All physics trees have the
same event count; HiForest metadata has one job entry. This is an older
jet-analyzer schema, not output from the updated slim producer.

| Tree/group | Existing branches | Prior slim target | Existing compressed branch bytes |
| --- | ---: | ---: | ---: |
| HLT | 1540 | 1 | 1,886,856 |
| Skim | 8 | 3 | 54,446 |
| Event | 12 | 8 | 751,124 |
| Tracks | 41 | 11 | 229,883,699 |
| Each R4/R8 jet tree | 47 each | 19 each | 840,866 / 2,425,221 |
| OO rho profile | 3 | 0 | 8,455,502 |
| Four random-cone trees | 5 each | 0 | 2,929,238 |
| HiForest metadata | 3 | 3 | 305 |

The track tree accounts for about 93% of the old file's compressed branch
payload. This table and the earlier 118,168,447 / 247,227,257 byte estimate
refer to the prior contract (19 OO jet branches and no stored scalar rho);
they must not be used as a size estimate for the updated OO contract. The new
44-branch OO jet trees and four-branch scalar rho tree need a fresh forest to
measure their storage.

The sample contains 4,348,262 tracks. Of those, 2,276,391 (52.35%) satisfy the
production retention pT/eta cuts. Quality filtering remains downstream.
All 1,587,988 tracks passing the nominal pT, eta, high-purity, relative-pT-error
and first-vertex significance cuts in this file survive the retention cuts.
This is a track-selection check, not a full EEC event-loop comparison.

Both uploaded jet trees lack `rawpt, trackMax, jtarea` and all ten
`jtPf*` composition branches. They contain older quantities such as
`jtptUncorrected` and `chargedMax`. The latter are not automatically renamed:
the current native input contract and measured area cannot be recovered by
pruning. Reforest with the updated analyzer for current Run-3 selection/JEC.
The older composition fields that are present remain retained in the new OO
contract.
The old OO sample also lacks the scalar rho tree; new OO forests will store
the same scalar as ppRef, although the current EEC L1FastJet option does not
apply to OO.

## Checking new files

A reusable checker enumerates every ROOT tree and branch, reports unwanted
output, validates the required names and entry counts, and optionally checks
the native rho/area representation and values. It requires uproot, awkward,
and numpy. For example, from the CMSSW src directory:

```bash
python3 HeavyIonsAnalysis/Configuration/test/check_eec_forest_content.py HeavyIonsAnalysis/Configuration/L1_object_data.root --sample oo-data --strict --inventory forest_inventory.csv
python3 HeavyIonsAnalysis/Configuration/test/check_eec_forest_content.py HeavyIonsAnalysis/Configuration/unembedded_L1HLT_objects_MC.root --sample oo-mc --strict
python3 HeavyIonsAnalysis/Configuration/test/check_eec_forest_content.py HiForestMiniAOD_ppref.root --sample pp-data --strict --l1
```

These changes include C++; run `scram b -j 4` and rebuild the companion EEC
analysis before running. The uploaded file was inspected with uproot, and
offline branch-contract/booking checks were run. Full CMSSW and EEC event-loop
execution is still required; no numerical reduction in newly produced file
size is claimed.

## Nebraska CRAB production

The OO data list `datasets_OO_DATA.txt` contains `IonPhysics0` through
`IonPhysics59`. The data submission script makes one CRAB task per dataset
and skips a request whose local CRAB project directory already exists.
All tasks use the OO slim data or MC pset and store TFileService output at
`T2_US_Nebraska` beneath `/store/user/jvelazqu/HiForest/OO2025/DATA`
or `/store/user/jvelazqu/HiForest/OO2025/MC`. The new
`Nebraska_slim_v1` request names separate these outputs from earlier runs.
The data task keeps the OO golden JSON mask. The current MC config uses the
`QCD-dijet_Pthat-15_TuneCP5_OO_5p36TeV_pythia8` MINIAODSIM dataset.

From `HeavyIonsAnalysis/Configuration` on lxplus, after updating this branch
and running `cmsenv` and `scram b -j 4`:

```bash
bash submit_multidataset_OO_DATA.sh --dry-run
bash submit_multidataset_OO_DATA.sh 2>&1 | tee submit_OO_DATA_Nebraska_slim.log

crab submit -c crab_OO_MC.py
crab status -d crab_projects/OO2025_MC_Nebraska_slim/crab_HiForest_OO2025_MC_Pythia8_Nebraska_slim_v1
```

To try one data stream first, put just its dataset name in a separate text
file and pass that file to the same script. The defaults are 50,000 events
per data job and 30,000 per MC job with `EventAwareLumiBased` splitting.
CRAB overrides the small interactive `process.maxEvents` values in these
psets. Check the CRAB status and job count before submitting all streams;
if a task exceeds CRAB's job limit, raise its `Data.unitsPerJob` and use
a new request version. No correction is applied by storing OO scalar rho.
