# Run 3 Geometries

The Run 3 geometry is automatically created using the script [generate2021Geometry.py](./scripts/generate2021Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T3: 2021 baseline after separating tracker specific material
* T4: as T3, but with zero material

Calorimeters:
* C1: 2021 baseline

Muon system:
* M1: 2021 baseline

PPS:
* P2: 2021 baseline (after using its own material files for pixel)

The script also handles the common and forward elements of the geometry:
* O3: 2021 baseline
* O4: as O3, but with zero material
* F1: 2021 baseline

Several detector combinations have been generated:
* 2021 = T3+C1+M1+P2+O3+F1
* 2021ZeroMaterial = T4+C1+M1+P2+O4+F1

# Phase 2 Geometries

The Phase 2 geometries are automatically created using the script [generate2026Geometry.py](./scripts/generate2026Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T15: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.3) (Active geometry: same as T14. Material Budget: major update in IT, gathering info from recent Mechanical designs.)
* T21: Phase2 tilted tracker. Outer Tracker (v8.0.0): TBPS update in Layer 1 (facilitate IT insertion) + In all TEDD, update sensors Z inter-spacing. Inner Tracker: (v6.1.5) from previous T17
(TFPX: Changed sensors spacing within all double-disks + Increased distance between Disks 6 and 7 + TBPX portcards between Disks 6 and 7.)
* T22: Phase2 tilted tracker. Outer Tracker (v8.0.0): same as T21. Inner Tracker: Based on (v6.1.5) (T21), but with 50x50 pixel aspect ratio everywhere.
* T23: Phase2 tilted tracker. Outer Tracker (v8.0.0): same as T21. Inner Tracker: Based on (v6.1.5) (T21), but with 3D sensors in TBPX L1 + TBPX L2 + TFPX R1.

Calorimeters:
* C9: HGCal (v11 post TDR HGCal Geometry w/ corner centering for HE part) + Phase2 HCAL and EB + Tracker cables
* C10: HGCal (as in C9) + HFNose with corrected wafer size + Phase2 HCAL and EB
* C11: HGCal (v12 post TDR HGCal Geometry same as C9 + modified support structure + full list of masked wafers)
* C12: HGCal (as in C11) + HFNose with corrected wafer size + Phase2 HCAL and EB
* C13: HGCal (reading the constants of the flat file and made provision to be used downstream) + Phase2 HCAL and EB
* C14: HGCal (reading the constants of the flat file and use it to create geometry) + Phase2 HCAL and EB

Muon system:
* M4: Phase2 muon system for TDR w/ GE2/1, ME0, RE3/1, RE4/1 (incl. granularity in ME0, staggered GE2/1), 96 iRPC strips, no overlaps, MB4Shields
* M6: same as M4 with right value for YE3 size, no "hidden" overlaps, iRPC updated, adjustment of ME0 in view of updated boundaries
* M7: same as M6 with further ajustment of ME0 for boundaries
* M8: same as M7 with changed number of strips for GE21
* M9: same as M8 with GE0 replacing ME0

Fast Timing system:
* I10: Fast Timing detector (LYSO barrel (bars along phi flat), silicon endcap), w/ passive materials, ETL in position defined in O4, material adjustments
* I11: Same as I10, xml reorganized, comparison base for new ETL and DD4hep migration
* I12: Starting from I11, new ETL layout from MTD TDR
* I13: Starting from I11, new ETL layout from post MTD TDR (2 sectors per disc face)

The script also handles the common and forward elements of the geometry:
* O4: detailed cavern description, changes for modified CALO region for endcap part, no overlaps inside the Muon System 
* O5: same as O4 but with changes needed for new support structure 
* O6: same as O5 with changes needed for new defintion of boundaries
* O7: same as O6 with changes needed for new defintion of calorimeter boundaries
* F2: modifications needed to accommodate detailed cavern, ZDC description is removed.
* F3: same as F2 but changes due to HFNose
* F4: same as F2 but with modifications needed to forward shield
* F5: same as F4 but changes due to HFNose
* F6: same as F4 with modifications needed for BRM and forward shield

Several detector combinations have been generated:
* D49 = T15+C9+M4+I10+O4+F2
* D50 = T15+C9+M4+I11+O4+F2
* D60 = T15+C10+M4+I10+O4+F3
* D64 = T22+C11+M4+I11+O5+F4
* D65 = T23+C11+M4+I11+O5+F4
* D66 = T21+C11+M8+I11+O5+F4
* D67 = T21+C11+M9+I11+O5+F4
* D68 = T21+C11+M6+I11+O5+F4
* D69 = T21+C12+M6+I11+O5+F5
* D70 = T21+C13+M7+I11+O6+F6
* D71 = T21+C14+M7+I11+O7+F6
* D72 = T21+C11+M6+I12+O5+F4
* D73 = T21+C11+M6+I13+O5+F4

D49 is the HLT TDR baseline.
