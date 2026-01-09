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
* For Phase-2, use [generateRun4Geometry.py](./scripts/generateRun4Geometry.py) and [dictRun4Geometry.py](./python/dictRun4Geometry.py) instead.
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
* M4: 2025 with additional chambers in GE21 and iRPC and modified DTShield
* M5: Same as M1 with modified RPC
* M6: Same as M2 with modified RPC
* M7: Same as M3 with modified RPC
* M8: Same as M4 with modified RPC
* M9: Same as M1 with modified RPC, corrected for phi staggering and z-position
* M10: Same as M2 with modified RPC, corrected for phi staggering and z-position
* M11: Same as M3 with modified RPC, corrected for phi staggering and z-position
* M12: Same as M4 with modified RPC, corrected for phi staggering and z-position
* M13: Same as M9 with modified DTShield
* M14: Same as M10 with modified DTShield
* M15: Same as M11 with modified DTShield
* M16: Same as M12 with unmounted GE11 for 2025
* M17: Same as M16 where the list of unmounted GE11 is correctd

PPS:
* P7: 2021 baseline (after removing overlaps and using common materials whenever possible)
* P8: First 2025 version with the rotated PPS detectors

The script also handles the common and forward elements of the geometry:
* O4: as O6, but with zero material
* O5: as O6, but with trackermaterial removed (they are in T5, T6, T7, T8)
* O6: 2021 baseline
* O7: 2021 with added material for muon shield
* O8: as O4 with added material for muon shield
* O9: as O5 with added material for muon shield
* O10: as O7 with the material for IP to ZDC or ZDC to beyond as vacuum
* F1: 2021 baseline
* F2: same as F1 with modified file zdc.xml from ZDC group
* F3: same as F2 with added simulti geometry for RPD
* F4: same as F3 with corrected version of zdc.xml from ZDC group
* F5: same as F4 with corrected diameter of the Fibre

Several detector combinations have been generated:
* 2021 = T3+C3+M13+P7+O7+F1
* 2021ZeroMaterial = T4+C1+M9+P7+O4+F1
* 2021FlatMinus05Percent = T5+C1+M9+P7+O5+F1
* 2021FlatMinus10Percent = T6+C1+M9+P7+O5+F1
* 2021FlatPlus05Percent = T7+C1+M9+P7+O5+F1
* 2021FlatPlus10Percent = T8+C1+M9+P7+O5+F1
* 2023 = T3+C2+M14+P7+O7+F3
* 2023ZeroMaterial = T4+C1+M10+P7+O4+F2
* 2023FlatMinus05Percent = T5+C1+M10+P7+O5+F2
* 2023FlatMinus10Percent = T6+C1+M10+P7+O5+F2
* 2023FlatPlus05Percent = T7+C1+M10+P7+O5+F2
* 2023FlatPlus10Percent = T8+C1+M10+P7+O5+F2
* 2024 = T3+C2+M15+P7+O7+F3
* 2024ZeroMaterial = T4+C2+M11+P7+O4+F2
* 2024FlatMinus05Percent = T5+C2+M11+P7+O5+F2
* 2024FlatMinus10Percent = T6+C2+M11+P7+O5+F2
* 2024FlatPlus05Percent = T7+C2+M11+P7+O5+F2
* 2024FlatPlus10Percent = T8+C2+M11+P7+O5+F2
* 2025 = T3+C2+M17+P8+O10+F5
* 2025ZeroMaterial = T4+C2+M12+P8+O8+F3
* 2025FlatMinus05Percent = T5+C2+M12+P8+O9+F3
* 2025FlatMinus10Percent = T6+C2+M12+P8+O9+F3
* 2025FlatPlus05Percent = T7+C2+M12+P8+O9+F3
* 2025FlatPlus10Percent = T8+C2+M12+P8+O9+F3

# Phase 2 Geometries

The Phase 2 geometries are automatically created using the script [generateRun4Geometry.py](./scripts/generateRun4Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T35: Phase2 tilted tracker. Outer Tracker (v8.0.0), Inner Tracker (v7.0.2): Based on (v6.1.5), but with (more realistic) 3D sensors in TBPX L1. The outer radius of the tracker volume is reduced to avoid a clash with the BTL geometry. Modified Tracker volume so that it touches CALO on the outer side and BeamPipe on the inner side
* T36: OT (v8.0.6): increased (smallDelta +300 micron) inter-ladder radial spacing TB2S. IT (v7.4.1): TBPX as in T35 with 0.4 mm gap between Z+ and Z-
* T37: OT (v8.0.6): increased (smallDelta +300 micron) inter-ladder radial spacing TB2S. IT (v7.4.2): TBPX as in T35 with 0.7+0.4+0.7 mm gap between Z+ and Z-
* T38: OT (v8.0.6): increased (smallDelta +300 micron) inter-ladder radial spacing TB2S. IT (v7.4.4): TBPX as in T35 with 1.3+0.4+1.3 mm gap between Z+ and Z-
* T39: Same as T35 but introducing BigPixels in InnerTracker (1x2 planar and 2x2 planar modules)

Calorimeters:
* C18: HGCal (v17 version of HGCal geometry created for a new flat file for silicon having 47 layers, ideas of cassettes, new orientation indices for full and partial wafers) + Phase2 HCAL and EB
* C19: HGCal (v17 version of HGCal geometry as in C18 but without internal cells in the Geant4 geometry definition) + Phase2 HCAL and EB
* C20: HGCal (v17 version of HGCal geometry as in C18) + HFNose with corrected wafer size + Phase2 HCAL and EB
* C22: HGCal (v18 version of HGCal geometry with calibration cells, nonzero cassette retraction, correct mousebite, guard ring, proper cell size) + Phase2 HCAL and EB
* C25: same as C18 but changing ebalgo.xml to make it more conformant with standard and removing overlaps
* C26: HGCal (v19 version of HGCal geometry with calibration cells, nonzero cassette retraction, correct mousebite, guard ring, proper cell size) + Phase2 HCAL and EB
* C27: HGCal (same as the v19 version which is in C26 but without internal cells in the Geant4 geometry definition) + Phase2 HCAL and EB
* C28: HGCal (v19 version of HGCal geometry as in C22 but turning off all dead areas and gaps) + Phase2 HCAL and EB

Muon system:
* M14: Phase2 muon system for TDR w/ GE2/1, GE0, RE3/1, RE4/1 (incl. granularity in ME0, staggered GE2/1), 96 iRPC strips; no overlaps, MB4Shields with right value for YE3 size and Shield structure modified in muonYoke, changed number of strips and corrected eta partition size for GE21, a realistic support structure for GE0 and adjustment for boundaries, right front-back relation between alternate phi segments
* M15: same as M14 but removing overlaps in yoke, MB3, GE0 + adding DT shield
* M16: same as M15 with reverting RPC endcap disk4 rotation

Fast Timing system:
* I17: Fast Timing detector (LYSO barrel (bars along phi flat), silicon endcap), w/ passive materials, material adjustments, new ETL layout from post MTD TDR (2 sectors per disc face), updated sensor structure, disc z location and passive materials, addition of notch and revision of envelope, revised BTL with complete passive material description, BTL with one crystal thickness (type) only, ETL with LGAD split into two sensors
* I18: Same as I17, needed for updated BTL numbering scheme and BTLDetId format
* I20: BTL I18/v4, ETL v10 with 2024 1.7 layout
* I21: BTL I18/v4, ETL v11 with 2024 full layout, same as v9 with additional level and id for service hybrids
* I22: BTL I18/v4, ETL v12 with 2024 1.7 layout, same as v10 with additional level and id for service hybrids

The script also handles the common and forward elements of the geometry:
*  O9: detailed cavern description, changes for modified CALO region for endcap part, no overlaps inside the Muon System, new support structure, newer definition of calorimeter boundaries, support the additional notch in ETL
* O10: same as O9 with changes needed to support the shields for DT

* F8: modifications needed to accommodate detailed cavern, forward shield, HFNose; ZDC description is removed.
* F9: same as F8 after removing overlap in rotated shield

Several detector combinations have been generated:
* D104 = T35+C22+M14+I16+O9+F8
* D110 = T35+C18+M14+I17+O9+F8
* D111 = T36+C24+M14+I17+O9+F8
* D112 = T37+C24+M14+I17+O9+F8
* D113 = T38+C24+M14+I17+O9+F8
* D114 = T39+C19+M14+I17+O9+F8
* D120 = T35+C26+M16+I20+O10+F9
* D121 = T35+C25+M16+I18+O10+F9  (Current Phase-2 baseline from CMSSW_15_1_0_pre4)
* D122 = T35+C27+M16+I18+O10+F9
* D123 = T35+C28+M16+I18+O10+F9
* D124 = T35+C25+M16+I21+O10+F9
* D125 = T35+C25+M16+I22+O10+F9
