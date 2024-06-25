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
* For Phase-2, use [generate2026Geometry.py](./scripts/generate2026Geometry.py) and [dict2026Geometry.py](./python/dict2026Geometry.py) instead.
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
* M1: 2021 baseline with additional chambers in GE21 and iRPC31/41
* M2: 2023 GE21 shifted in position
* M3: 2024 with additional chambers in GE21 and iRPC31

PPS:
* P7: 2021 baseline (after removing overlaps and using common materials whenever possible)

The script also handles the common and forward elements of the geometry:
* O4: as O6, but with zero material
* O5: as O6, but with trackermaterial removed (they are in T5, T6, T7, T8)
* O6: 2021 baseline
* F1: 2021 baseline
* F2: same as F1 with modified file zdc.xmlfrom ZDC group
* F3: same as F2 with added simulti geometry for RPD

Several detector combinations have been generated:
* 2021 = T3+C1+M1+P7+O6+F1
* 2021ZeroMaterial = T4+C1+M1+P7+O4+F1
* 2021FlatMinus05Percent = T5+C1+M1+P7+O5+F1
* 2021FlatMinus10Percent = T6+C1+M1+P7+O5+F1
* 2021FlatPlus05Percent = T7+C1+M1+P7+O5+F1
* 2021FlatPlus10Percent = T8+C1+M1+P7+O5+F1
* 2023 = T3+C1+M2+P7+O6+F3
* 2023ZeroMaterial = T4+C1+M2+P7+O4+F2
* 2023FlatMinus05Percent = T5+C1+M2+P7+O5+F2
* 2023FlatMinus10Percent = T6+C1+M2+P7+O5+F2
* 2023FlatPlus05Percent = T7+C1+M2+P7+O5+F2
* 2023FlatPlus10Percent = T8+C1+M2+P7+O5+F2
* 2024 = T3+C1+M3+P7+O6+F3
* 2024ZeroMaterial = T4+C1+M3+P7+O4+F2
* 2024FlatMinus05Percent = T5+C1+M3+P7+O5+F2
* 2024FlatMinus10Percent = T6+C1+M3+P7+O5+F2
* 2024FlatPlus05Percent = T7+C1+M3+P7+O5+F2
* 2024FlatPlus10Percent = T8+C1+M3+P7+O5+F2

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
* T30: Phase2 tilted tracker. Exploratory geometry *only to be used in D91 for now*. Outer Tracker (v8.0.1): based on v8.0.0 with updated TB2S spacing. Inner Tracker (v6.4.0): based on v6.1.5 but TFPX with more realistic module positions.
* T31: Phase2 tilted tracker. The tracker description is identical to T24/T21. The outer radius of the tracker volume is reduced to avoid a clash with the BTL geometry. The positions of the tracker components are not affected
* T32: Phase2 tilted tracker. The tracker description is identical to T25. The outer radius of the tracker volume is reduced to avoid a clash with the BTL geometry (same as T31). The positions of the tracker components are not affected. This geometry is intended as a transition step towards a realistic configuration with 3D sensors in TBPX layer1.
* T33: Phase2 tilted tracker. Identical to T32 apart from a more realistic description of the 3D sensors in TBPX layer1.
* T34: Same as T32 with the exception of modified Tracker volume so that it touches CALO on the outer side and BeamPipe on the inner side
* T35: Same as T33 with the exception of modified Tracker volume so that it touches CALO on the outer side and BeamPipe on the inner side
* T36: OT (v8.0.6): increased (smallDelta +300 micron) inter-ladder radial spacing TB2S. IT (v7.4.1): TBPX as in T35 with 0.4 mm gap between Z+ and Z-
* T37: OT (v8.0.6): increased (smallDelta +300 micron) inter-ladder radial spacing TB2S. IT (v7.4.2): TBPX as in T35 with 0.7+0.4+0.7 mm gap between Z+ and Z-
* T38: OT (v8.0.6): increased (smallDelta +300 micron) inter-ladder radial spacing TB2S. IT (v7.4.4): TBPX as in T35 with 1.3+0.4+1.3 mm gap between Z+ and Z-
* T39: Same as T35 but introducing BigPixels in InnerTracker (1x2 planar and 2x2 planar modules)

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
* C20: HGCal (v17 version of HGCal geometry as in C18) + HFNose with corrected wafer size + Phase2 HCAL and EB (used in 2026D93)
* C21: HGCal (v17 version of HGCal geometry as in C19 but turning off all dead areas and gaps) + Phase2 HCAL and EB (used in 2026D101)
* C22: HGCal (v18 version of HGCal geometry as in C18 with calibration cells, nonzero cssette retraction, correct mousebite, guard ring, proper cell size) + Phase2 HCAL and EB (used in 2026D104)
* C23: HGCal (same as the v18 version which is in C22 but without internal cells in the Geant4 geometry defintiion) + Phase2 HCAL and EB (used in 2026D106)
* C24: HGCal (v18 version of HGCal geometry as in C122 but turning off all dead areas and gaps) + Phase2 HCAL and EB (used in 2026D109)

Muon system:
* M4: Phase2 muon system for TDR w/ GE2/1, ME0, RE3/1, RE4/1 (incl. granularity in ME0, staggered GE2/1), 96 iRPC strips, no overlaps, MB4Shields
* M6: same as M4 with right value for YE3 size, no "hidden" overlaps, iRPC updated, adjustment of ME0 in view of updated boundaries
* M7: same as M6 with further ajustment of ME0 for boundaries
* M8: same as M7 with changed number of strips for GE21
* M9: same as M8 with GE0 replacing ME0
* M10: same as M9 but with a realistic support structure for GE0, Shield structure modified in muonYoke
* M11: same as M10 but with a corrected eta partition sizes for GE21

Fast Timing system:
* I10: Fast Timing detector (LYSO barrel (bars along phi flat), silicon endcap), w/ passive materials, ETL in position defined in O4, material adjustments
* I11: Same as I10, xml reorganized, comparison base for new ETL and DD4hep migration
* I12: Starting from I11, new ETL layout from MTD TDR
* I13: Starting from I11, new ETL layout from post MTD TDR (2 sectors per disc face)
* I14: Same as I13, updated sensor structure, disc z location and passive materials
* I15: Same as I14, addition of notch and revision of envelope
* I16: Starting from I15, revised BTL with complete passive material description, it needs Tracker T31 or newer
* I17: Same as I16, BTL with one crystal thickness (type) only, ETL with LGAD split into two sensors

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
* D86 = T24+C17+M10+I14+O8+F8
* D88 = T24+C17+M10+I15+O9+F8
* D91 = T30+C17+M10+I15+O9+F8
* D92 = T24+C18+M10+I15+O9+F8
* D93 = T24+C19+M10+I15+O9+F8
* D94 = T24+C20+M10+I15+O9+F8
* D95 = T31+C17+M10+I16+O9+F8
* D96 = T31+C18+M10+I16+O9+F8
* D97 = T25+C17+M10+I15+O9+F8
* D98 = T32+C17+M10+I16+O9+F8 (Current Phase-2 baseline from CMSSW_13_2_0_pre2)
* D99 = T32+C18+M10+I16+O9+F8
* D100 = T34+C17+M11+I16+O9+F8
* D101 = T34+C18+M11+I16+O9+F8
* D102 = T35+C17+M11+I16+O9+F8
* D103 = T35+C21+M11+I17+O9+F8
* D104 = T35+C22+M11+I16+O9+F8
* D105 = T35+C17+M11+I17+O9+F8
* D106 = T35+C23+M11+I17+O9+F8
* D107 = T32+C17+M11+I17+O9+F8
* D108 = T35+C19+M11+I17+O9+F8
* D109 = T35+C24+M11+I17+O9+F8
* D110 = T35+C18+M11+I17+O9+F8
* D111 = T36+C24+M11+I17+O9+F8
* D112 = T37+C24+M11+I17+O9+F8
* D113 = T38+C24+M11+I17+O9+F8
* D114 = T39+C19+M11+I17+O9+F8

