The Run 3 geometry is automatically created using the script [generate2021Geometry.py](./scripts/generate2021Geometry.py).

The Phase2 geometries are automatically created using the script [generate2026Geometry.py](./scripts/generate2026Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T15: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.3) (Active geometry: same as T14. Material Budget: major update in IT, gathering info from recent Mechanical designs.)
* T17: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.5) TFPX: Changed sensors spacing within all double-disks + Increased distance between Disks 6 and 7 + Put TBPX portcards between Disks 6 and 7.
* T19: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v7.0.0) Inner Tracker description with 3D sensors in TBPX L1 + TBPX L2 + TFPX R1.
* T20: Phase2 tilted tracker. Outer Tracker (v6.1.6): All sensors 200 um -> 290 um + Update in Module MB + PS modules: s-sensor 164 um longer + Major update in OTST MB. Inner Tracker: (v6.1.5) from T17 is called.
* T21: Phase2 tilted tracker. Outer Tracker (v8.0.0): TBPS update in Layer 1 (facilitate IT insertion) + In all TEDD, update sensors Z inter-spacing. Inner Tracker: (v6.1.5) from T17.
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
* M5: same as M4 but with: the right value for YE3 size, no "hidden" overlaps inside the Muon System and iRPC updated.
* M6: same as M5 but with adjustment of ME0 in view of updated boundaries
* M7: same as M6 with further ajustment of ME0 for boundaries

Fast Timing system:
* I10: Fast Timing detector (LYSO barrel (bars along phi flat), silicon endcap), w/ passive materials, ETL in position defined in O4, material adjustments
* I11: Same as I10, xml reorganized, comparison base for new ETL and DD4hep migration
* I12: Starting from I11, new ETL layout from MTD TDR

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
* D51 = T17+C9+M4+I10+O4+F2
* D53 = T15+C9+M4+I12+O4+F2
* D54 = T19+C9+M4+I10+O4+F2
* D56 = T20+C9+M4+I10+O4+F2
* D57 = T17+C11+M6+I11+O5+F4
* D58 = T17+C12+M6+I11+O5+F5
* D59 = T17+C13+M7+I11+O6+F6
* D60 = T15+C10+M4+I10+O4+F3
* D61 = T17+C9+M5+I10+O4+F2
* D62 = T17+C14+M7+I11+O7+F6
* D63 = T21+C11+M4+I11+O5+F4
* D64 = T22+C11+M4+I11+O5+F4
* D65 = T23+C11+M4+I11+O5+F4

D49 is the HLT TDR baseline.
