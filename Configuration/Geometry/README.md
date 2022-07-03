# To work on geometry package

### To create or update geometries
```
git cms-addpkg Geometry/CMSCommonData
git cms-addpkg Configuration/Geometry
scram b -j 8
cd Configuration/Geometry
vi python/dict2021Geometry.py
python3 ./scripts/generate2021Geometry.py -D 2021
```
Note:
* For Phase-2, use [generate2026Geometry.py](./python/dict2026Geometry.py) and [generate2026Geometry.py](./scripts/generate2026Geometry.py) instead.
* For the list of geometries, see below.

# Run 3 Geometries

The Run 3 geometry is automatically created using the script [generate2021Geometry.py](./scripts/generate2021Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T3: 2021 baseline after separating tracker specific material
* T4: as T3, but with zero material
* T5: as T3, but with tracker material budget reduced by 5%
* T6: as T3, but with tracker material budget reduced by 10%
* T7: as T3, but with tracker material budget increased by 5%
* T8: as T3, but with tracker material budget increased by 10%

Calorimeters:
* C1: 2021 baseline

Muon system:
* M1: 2021 baseline

PPS:
* P7: 2021 baseline (after removing overlaps and using common materials whenever possible)

The script also handles the common and forward elements of the geometry:
* O4: as O6, but with zero material
* O5: as O6, but with trackermaterial removed (they are in T5, T6, T7, T8)
* O6: 2021 baseline
* F1: 2021 baseline

Several detector combinations have been generated:
* 2021 = T3+C1+M1+P7+O6+F1
* 2021ZeroMaterial = T4+C1+M1+P7+O4+F1
* 2021FlatMinus05Percent = T5+C1+M1+P7+O5+F1
* 2021FlatMinus10Percent = T6+C1+M1+P7+O5+F1
* 2021FlatPlus05Percent = T7+C1+M1+P7+O5+F1
* 2021FlatPlus10Percent = T8+C1+M1+P7+O5+F1

# Phase 2 Geometries

The Phase 2 geometries are automatically created using the script [generate2026Geometry.py](./scripts/generate2026Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T15: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.3) (Active geometry: same as T14. Material Budget: major update in IT, gathering info from recent Mechanical designs.)
* T21: Phase2 tilted tracker. Outer Tracker (v8.0.0): TBPS update in Layer 1 (facilitate IT insertion) + In all TEDD, update sensors Z inter-spacing. Inner Tracker: (v6.1.5) from previous T17
(TFPX: Changed sensors spacing within all double-disks + Increased distance between Disks 6 and 7 + TBPX portcards between Disks 6 and 7.)
* T24: Phase2 tilted tracker. Tracker detector description itself is identical to T21 (OT800 IT615). Change of paradigm, entire description reworked to be compatible with DD4hep library.
* T25: Phase2 tilted tracker. Outer Tracker (v8.0.0): same as T24/T21. Inner Tracker (v7.0.2): Based on (v6.1.5) (T24/T21), but with 3D sensors in TBPX L1. Compatible with DD4hep library.
* T26: Phase2 tilted tracker. Outer Tracker (v8.0.0): same as T24/T21. Inner Tracker (v7.0.3): Based on (v6.1.5) (T24/T21), but with 3D sensors in TBPX L1 and 50x50 pixel aspect ratio in TFPX and TEPX. Compatible with DD4hep library.
* T30: Phase2 titled tracker. Exploratory geometry *only to be used in D91 for now*. Outer Tracker (v8.0.1): based on v8.0.0 with updated TB2S spacing. Inner Tracker (v6.4.0): based on v6.1.5 but TFPX with more realistic module positions.

Calorimeters:
* C9: HGCal (v11 post TDR HGCal Geometry w/ corner centering for HE part) + Phase2 HCAL and EB + Tracker cables (used in 2026D49)
* C10: HGCal (as in C9) + HFNose with corrected wafer size + Phase2 HCAL and EB (used in 2026D60)
* C11: HGCal (v12 post TDR HGCal Geometry same as C9 + modified support structure + full list of masked wafers) + Phase2 HCAL and EB + Tracker cables (used in 2026D68)
* C13: HGCal (v13 version which reads the input from the flat file, uses these for checks and makes provision to be used downstream) + Phase2 HCAL and EB (used in 2026D70, 2026D84)
* C14: HGCal (v14 version reading the input from the flat file and uses it to create geometry, still using masking to define partial wafers) + Phase2 HCAL and EB (used in 2026D76-81, 2026D85, 2026D87)
* C15: HGCal (as in C14) + HFNose with corrected wafer size  + Phase2 HCAL and EB (used in 2026D82)
* C16: HGCal (v15 version of HGCal geometry created using real full and partial silicon modules using the constants of the flat file) + Phase2 HCAL and EB (used in 2026D83)
* C17: HGCal (v16 version of HGCal geometry created with new longitudinal structure having 47 layers and new definition of partial wafers iusing the constants of the flat file) + Phase2 HCAL and EB (used in 2026D86, 2025D88)
* C18: HGCal (v17 version of HGCal geometry created for a new flat file for silicon having 47 layers, ideas of cassettes, new orientation indices for full and partial wafers) + Phase2 HCAL and EB (used in 2026D92)
* C19: HGCal (v17 version of HGCal geometry as in C18 but without internal cells in the Geant4 geometry definition) + Phase2 HCAL and EB (used in 2026D93)

Muon system:
* M4: Phase2 muon system for TDR w/ GE2/1, ME0, RE3/1, RE4/1 (incl. granularity in ME0, staggered GE2/1), 96 iRPC strips, no overlaps, MB4Shields
* M6: same as M4 with right value for YE3 size, no "hidden" overlaps, iRPC updated, adjustment of ME0 in view of updated boundaries
* M7: same as M6 with further ajustment of ME0 for boundaries
* M8: same as M7 with changed number of strips for GE21
* M9: same as M8 with GE0 replacing ME0
* M10: same as M9 but with a realistic support structure for GE0, Shield structure modified in muonYoke

Fast Timing system:
* I10: Fast Timing detector (LYSO barrel (bars along phi flat), silicon endcap), w/ passive materials, ETL in position defined in O4, material adjustments
* I11: Same as I10, xml reorganized, comparison base for new ETL and DD4hep migration
* I12: Starting from I11, new ETL layout from MTD TDR
* I13: Starting from I11, new ETL layout from post MTD TDR (2 sectors per disc face)
* I14: Same as I13, updated sensor structure, disc z location and passive materials
* I15: Same as I14, addition of notch and revision of envelope

The script also handles the common and forward elements of the geometry:
* O4: detailed cavern description, changes for modified CALO region for endcap part, no overlaps inside the Muon System 
* O5: same as O4 but with changes needed for new support structure 
* O6: same as O5 with changes needed for new defintion of boundaries
* O7: same as O6 with changes needed for new defintion of calorimeter boundaries
* O8: same as O7 with changes needed for a newer definition of calorimeter boundaries
* O9: same as O8 with changes needed to support the additional notch in ETL

* F2: modifications needed to accommodate detailed cavern, ZDC description is removed.
* F3: same as F2 but changes due to HFNose
* F4: same as F2 but with modifications needed to forward shield
* F5: same as F4 but changes due to HFNose
* F6: same as F4 with modifications needed for BRM and forward shield
* F7: same as F6 with modifications needed for HFNose
* F8: same as F6 or F7 without BRM

Several detector combinations have been generated:
* D49 = T15+C9+M4+I10+O4+F2 (HLT TDR baseline)
* D60 = T15+C10+M4+I10+O4+F3 (With HFNose)
* D68 = T21+C11+M6+I11+O5+F4 (For HGCAL study on evolution of detector)
* D70 = T21+C13+M7+I11+O6+F6 (For HGCAL study on evolution of detector)
* D76 = T21+C14+M9+I13+O7+F6
* D77 = T24+C14+M9+I13+O7+F6 (Current default scenario)
* D80 = T25+C14+M9+I13+O7+F6
* D81 = T26+C14+M9+I13+O7+F6
* D82 = T21+C15+M9+I13+O7+F7
* D83 = T24+C16+M9+I13+O7+F6
* D84 = T24+C13+M7+I11+O6+F6 (For HGCAL study on evolution of HGCal replacing D70)
* D85 = T24+C14+M9+I14+O7+F6
* D86 = T24+C17+M10+I14+O8+F6
* D88 = T24+C17+M10+I15+O9+F6
* D91 = T30+C17+M10+I15+O9+F6
* D92 = T24+C18+M10+I15+O9+F6
* D93 = T24+C19+M10+I15+O9+F6
