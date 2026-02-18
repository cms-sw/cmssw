# Scouting to MiniAOD/NanoAOD Conversion

This package converts Run3 scouting data to MiniAOD-compatible format, enabling NanoAOD production from scouting data.

## Overview

```
Scouting RAW  →  Scouting MiniAOD  →  Scouting NanoAOD
                 (Step 1: PAT)        (Step 2: NANO)
```

The workflow produces NanoAOD with standard branch naming (Muon_pt, Jet_eta, etc.) from scouting data, enabling analysis with standard NanoAOD tools.

## Quick Start

### Two-Step Production Workflow with cmsDriver

**Step 1: Scouting HLTSCOUT → MiniAOD**
```bash
cmsDriver.py scoutMini \
    --scenario pp \
    --conditions auto:run3_data_prompt \
    --era Run3_2024 \
    --eventcontent MINIAOD \
    --datatier MINIAOD \
    --step PAT:@Scout \
    --filein file:scouting.root \
    --fileout file:scoutingMiniAOD.root \
    --data -n 100
```

**Step 2: MiniAOD → NanoAOD**

Use our custom NanoAOD tables that read from PAT objects:
```bash
cmsDriver.py scoutNano \
    --scenario pp \
    --conditions auto:run3_data_prompt \
    --era Run3_2024 \
    --eventcontent NANOAOD \
    --datatier NANOAOD \
    --step NANO:@ScoutMini \
    --filein file:scoutingMiniAOD.root \
    --fileout file:scoutingNanoAOD.root \
    --processName NANO \
    --data -n 100
```

Note:
- Use `--era Run3_2025` for 2025 data
- The `--processName NANO` avoids conflicts with the MiniAOD process name
- `@Scout` and `@ScoutMini` flavors are defined in `autoPAT.py`
- For 2024+ data, the muon producer automatically switches to `hltScoutingMuonPackerVtx` via the `run3_scouting_2024` era modifier

### Test Configurations (for development)

```bash
# Step 1 test
cmsRun PhysicsTools/PatFromScouting/test/test_scoutingToMiniAOD_cfg.py

# Step 2 test
cmsRun PhysicsTools/PatFromScouting/test/test_scoutingNanoAOD_cfg.py

# Standard NanoAOD on scouting MiniAOD (with customizations)
cmsRun PhysicsTools/PatFromScouting/test/test_standardNanoAOD_cfg.py
```

## MiniAOD Collections

| Collection Name | Type | Description |
|-----------------|------|-------------|
| `packedPFCandidates` | `pat::PackedCandidateCollection` | PF candidates from scouting particles |
| `offlineSlimmedPrimaryVertices` | `reco::VertexCollection` | Primary vertices |
| `slimmedMuons` | `pat::MuonCollection` | Muons with isolation and track info |
| `slimmedElectrons` | `pat::ElectronCollection` | Electrons with shower shape in userFloats |
| `slimmedPhotons` | `pat::PhotonCollection` | Photons with shower shape in userFloats |
| `slimmedJets` | `pat::JetCollection` | AK4 PF jets with b-tagging |
| `slimmedMETs` | `pat::METCollection` | MET from scouting PF data |
| `scoutingTracks` | `reco::TrackCollection` | Tracks with hit info as ValueMaps |
| `offlineBeamSpot` | `reco::BeamSpot` | From conditions database |
| `fixedGridRhoFastjetAll` | `double` | Rho copied from HLT scouting |
| `gtStage2Digis` | `GlobalAlgBlkBxCollection` | L1 trigger decisions (unpacked from raw) |
| `gmtStage2Digis` | `l1t::MuonBxCollection` | L1 muons (copied from gtStage2Digis) |
| `caloStage2Digis` | `l1t::Jet/EGamma/Tau/EtSumBxCollection` | L1 calo objects (copied from gtStage2Digis) |

## NanoAOD Tables

| Category | Key Variables |
|----------|---------------|
| **Muon** | pt, eta, phi, mass, charge, pdgId, dxy, dz, trkChi2, nValidHits, isGlobal, isTracker, isStandalone, isPF, nStations, nTrackerLayers, nPixelLayers, nChambers, nChambersCSCorDT, nValidMuonHits, nValidPixelHits, nValidStripHits, relIso, ecalIso, hcalIso, trkIso, tkRelIso |
| **Electron** | pt, eta, phi, mass, charge, pdgId, dxy, dz, sieie, hoe, dEtaIn, dPhiIn, pfRelIso03_all, ecalIso, hcalIso, trkIso |
| **Photon** | pt, eta, phi, mass, sieie, hoe, ecalIso, hcalIso |
| **Jet** | pt, eta, phi, mass, area, chHEF, neHEF, chEmEF, neEmEF, muEF, nConstituents, chMultiplicity, neMultiplicity, btagCSV, btagDeepB |
| **MET** | pt, phi, sumEt |
| **PV** | x, y, z, xErr, yErr, zErr, chi2, ndof |
| **Event** | fixedGridRhoFastjetAll |
| **HLT** | All HLT trigger decisions (HLT_*, DST_*) |
| **L1** | All L1 trigger seeds (L1_*) |

## What's Available vs. Standard MiniAOD/NanoAOD

### Available (Full Support)

| Feature | Notes |
|---------|-------|
| Muon kinematics | pt, eta, phi, mass, charge |
| Muon track quality | chi2, nValidHits, dxy, dz |
| Muon ID flags | isGlobal, isTracker, isStandalone, isPF |
| Muon isolation | ecalIso, hcalIso, trkIso (calorimeter-based, not PF components) |
| Muon station/layer counts | nStations, nTrackerLayers, nPixelLayers, nChambers |
| Muon hit counts | nValidMuonHits, nValidPixelHits, nValidStripHits |
| Electron kinematics | pt, eta, phi, mass, charge |
| Electron shower shape | sigmaIetaIeta, H/E, dEtaIn, dPhiIn |
| Electron isolation | ecalIso, hcalIso, trackIso, PF relIso |
| Photon kinematics | pt, eta, phi |
| Photon shower shape | sigmaIetaIeta, H/E |
| Photon isolation | ecalIso, hcalIso |
| Jet kinematics | pt, eta, phi, mass, area |
| Jet composition | Energy fractions (chHEF, neHEF, etc.), multiplicities |
| Jet b-tagging | CSV, DeepCSV (pre-computed at HLT) |
| MET | pt, phi, sumEt |
| Primary vertices | Position, errors, chi2, ndof |
| HLT triggers | All trigger decisions as booleans |
| L1 triggers | All L1 seeds as booleans |
| L1 objects | Muons, jets, EGamma, taus, EtSum (via gmtStage2Digis/caloStage2Digis) |
| Pileup density | fixedGridRhoFastjetAll from HLT |
| Beam spot | From conditions database |

### Not Available (Scouting Limitations)

| Feature | Reason |
|---------|--------|
| Jet constituents | Not stored in scouting format |
| Pileup Jet ID | Requires iterating over jet daughters |
| Deep flavor b-tagging | Requires track-level constituent info |
| ParticleNet/DeepJet | Requires full constituent info |
| AK8/Fat jets | Not produced in scouting |
| Tau leptons | Not reconstructed in scouting |
| HLT trigger objects | hltTriggerSummaryAOD not saved in scouting |
| GenParticles | Data only (no MC scouting) |
| MET corrections | No Type-1 corrections available (raw MET only) |
| Electron/Photon MVA ID | Requires full shower info |

### Partially Available

| Feature | What Works | What Doesn't |
|---------|------------|--------------|
| Electron track | d0, dz, chi2 | No GSF track reference |
| Photon R9 | Stored if available | May be 0 for some photons |
| Jet corrections | Jets already corrected at HLT | No JEC uncertainties |

## Data Mapping Details

### Muons
- Uses `pat::Muon(const Run3ScoutingMuon&)` constructor from DataFormats/PatCandidates
- Track info embedded directly (fix applied for persistence)
- Isolation mapped to `reco::MuonIsolation` structure (ecalIso, hcalIso, trkIso)
- Station and hit count information derived from scouting muon data

### Electrons
Variables stored as userFloats (direct PAT electron construction requires detector references):
```
sigmaIetaIeta, hOverE, r9, sMin, sMaj, seedId
dEtaIn, dPhiIn, ooEMOop, missingHits
trkd0, trkdz, trkpt, trketa, trkphi, trkchi2overndf
rawEnergy, preshowerEnergy, corrEcalEnergyError, trackfbrem
ecalIso, hcalIso, trackIso
```

### Photons
Variables stored as userFloats:
```
sigmaIetaIeta, hOverE, r9, sMin, sMaj, seedId
rawEnergy, preshowerEnergy, corrEcalEnergyError
ecalIso, hcalIso, trkIso
```

### Jets
- Full `reco::PFJet::Specific` populated (energy fractions, multiplicities)
- B-discriminators: `pfCombinedSecondaryVertexV2BJetTags` (csv), `pfDeepCSVJetTags:probb` (mva)
- Jet area preserved

### Vertices
- Uses `pat::makeRecoVertex()` helper from DataFormats/PatCandidates
- Only valid vertices included (`isValidVtx() == true`)
- Full covariance matrix preserved

### MET
- Uses precomputed MET from scouting data (pfMetPt, pfMetPhi)

### Tracks
- Uses `pat::makeRecoTrack()` helper
- Hit pattern info stored as ValueMaps (nValidPixelHits, nValidStripHits, nTrackerLayersWithMeasurement)

### L1 Objects
- `gtStage2Digis`: L1 trigger decisions unpacked from raw FED data (`hltFEDSelectorL1`)
- `gmtStage2Digis`: L1 muons copied from gtStage2Digis Muon collection
- `caloStage2Digis`: L1 jets, EGamma, taus, EtSum copied from gtStage2Digis

## File Structure

```
PhysicsTools/PatFromScouting/
├── plugins/
│   ├── BuildFile.xml
│   ├── PatFromScoutingMuonProducer.cc
│   ├── PatFromScoutingElectronProducer.cc
│   ├── PatFromScoutingPhotonProducer.cc
│   ├── PatFromScoutingJetProducer.cc
│   ├── Run3ScoutingVertexToRecoVertexProducer.cc
│   ├── Run3ScoutingMETProducer.cc
│   ├── Run3ScoutingTrackToRecoTrackProducer.cc
│   ├── Run3ScoutingParticleToPackedCandidateProducer.cc
│   ├── Run3ScoutingL1MuonProducer.cc
│   ├── Run3ScoutingL1CaloProducer.cc
│   └── ScoutingRhoProducer.cc
├── python/
│   ├── autoPAT.py                      # cmsDriver flavor definitions (@Scout, @ScoutMini)
│   ├── scoutingToMiniAOD_cff.py        # Step 1: RAW → MiniAOD producers
│   ├── scoutingMiniAOD_cff.py          # Alternative MiniAOD config with rho variants
│   ├── scoutingNanoAOD_cff.py          # Step 2: MiniAOD → NanoAOD tables
│   ├── nanoAOD_scouting_cff.py         # Customizations for standard NanoAOD on scouting MiniAOD
│   └── nanoAOD_customizations_cff.py   # Additional NanoAOD customization helpers
├── test/
│   ├── BuildFile.xml
│   ├── test_scoutingToMiniAOD_cfg.py   # Step 1 test config
│   ├── test_scoutingNanoAOD_cfg.py     # Step 2 test config
│   ├── test_standardNanoAOD_cfg.py     # Standard NanoAOD on scouting MiniAOD test
│   ├── test_catch2_PatFromScouting.cc  # Unit tests
│   └── test_catch2_main.cc             # Catch2 test main
└── README.md
```

## Python Configuration Reference

### autoPAT.py

Defines `@`-prefixed flavor mappings for `cmsDriver --step PAT:@... / NANO:@...`:

| Flavor | Step | Description |
|--------|------|-------------|
| `@Scout` | PAT | Scouting HLTSCOUT → MiniAOD |
| `@ScoutVtx` | PAT | Same as `@Scout` but with vertex-aware muons (2024+) |
| `@ScoutMini` | NANO | Scouting MiniAOD → NanoAOD |

### scoutingToMiniAOD_cff.py

Producers for Step 1 (Scouting RAW → MiniAOD):

```python
from PhysicsTools.PatFromScouting.scoutingToMiniAOD_cff import scoutingToMiniAODTask

# Task includes all producers with standard MiniAOD names:
# packedPFCandidates, offlineSlimmedPrimaryVertices, offlineBeamSpot,
# slimmedMuons, slimmedElectrons, slimmedPhotons, slimmedJets, slimmedMETs,
# scoutingTracks, fixedGridRhoFastjetAll, gtStage2Digis, gmtStage2Digis, caloStage2Digis
```

### scoutingNanoAOD_cff.py

NanoAOD tables for Step 2:

```python
from PhysicsTools.PatFromScouting.scoutingNanoAOD_cff import customiseScoutingNanoAOD
process = customiseScoutingNanoAOD(process)
```

### nanoAOD_scouting_cff.py

Customizations for running **standard NanoAOD** on scouting MiniAOD (disables modules that require unavailable information):

```python
from PhysicsTools.PatFromScouting.nanoAOD_scouting_cff import customiseNanoForScoutingMiniAOD
process = customiseNanoForScoutingMiniAOD(process)
```

## Input Tags (HLT Defaults)

| Producer | Default Input Tag |
|----------|-------------------|
| PF Candidates | `hltScoutingPFPacker` |
| Vertices | `hltScoutingPrimaryVertexPacker:primaryVtx` |
| Muons | `hltScoutingMuonPacker` (2024+: `hltScoutingMuonPackerVtx` via era modifier) |
| Electrons | `hltScoutingEgammaPacker` |
| Photons | `hltScoutingEgammaPacker` |
| Jets | `hltScoutingPFPacker` |
| Tracks | `hltScoutingTrackPacker` |
| L1 Raw | `hltFEDSelectorL1` |
| Rho | `hltScoutingPFPacker:rho` |

## Dependencies

- `DataFormats/PatCandidates`
- `DataFormats/Scouting`
- `DataFormats/METReco`
- `DataFormats/JetReco`
- `DataFormats/VertexReco`
- `DataFormats/TrackReco`
- `DataFormats/MuonReco`
- `PhysicsTools/NanoAOD`
- `L1Trigger/L1TGlobal` (for L1TRawToDigi)

## Testing

```bash
# Build
cd $CMSSW_BASE/src
scram b -j8

# Run unit tests
scram b runtests

# Run full workflow
cmsRun PhysicsTools/PatFromScouting/test/test_scoutingToMiniAOD_cfg.py
cmsRun PhysicsTools/PatFromScouting/test/test_scoutingNanoAOD_cfg.py

# Verify output
python3 -c "
import ROOT
f = ROOT.TFile.Open('scoutingNanoAOD_test.root')
t = f.Get('Events')
print(f'Events: {t.GetEntries()}, Branches: {len(t.GetListOfBranches())}')
"
```

## Known Issues and Workarounds

### Standard NanoAOD Compatibility

Running standard NanoAOD (`cmsDriver --step NANO`) on scouting MiniAOD requires the customizations in `nanoAOD_scouting_cff.py` because:

1. **Jet update chains** - Standard NanoAOD tries to recompute jets from constituents
2. **Deep taggers** - Require constituent-level information
3. **Tau reconstruction** - Not available in scouting
4. **PPS protons** - Not in scouting data

The custom scouting NanoAOD (`NANO:@ScoutMini`) bypasses these issues by producing tables directly from the scouting MiniAOD collections.

### Muon Persistence Fix

A fix was applied to `DataFormats/PatCandidates/src/Muon.cc` to enable persistence of `pat::Muon` objects created from `Run3ScoutingMuon`. The original constructor created a `TrackRef` to a temporary local vector which could not be serialized.

## References

- [Run3 Scouting Data Formats](https://github.com/cms-sw/cmssw/tree/master/DataFormats/Scouting)
- [PAT Scouting Helpers](https://github.com/cms-sw/cmssw/blob/master/DataFormats/PatCandidates/interface/ScoutingDataHandling.h)
- [PhysicsTools/Scouting](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/Scouting) - Existing scouting tools
