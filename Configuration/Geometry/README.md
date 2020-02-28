The Phase2 geometries are automatically created using the script [generate2026Geometry.py](./scripts/generate2026Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T5: Phase2 tilted tracker (v6.1.3) w/ phase 2 pixel (v4.0.2.5) 
* T6: Phase2 tilted tracker (v6.1.4) w/ phase 2 pixel (v4.0.4) (TEDD slighly rotated + Inner Tracker barrel has lower radii than TDR T5 geometry)
* T14: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.3) (Based from T12. OT: reduced envelope. IT: new chip size, different radii, 2x2 modules everywhere in TEPX, new ring paradigm in TEPX)
* T15: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.3) (Active geometry: same as T14. Material Budget: major update in IT, gathering info from recent Mechanical designs.)
* T16: Active geometry: skewed Inner Tracker geometry. Material Budget: same as T15.
* T17: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.5) TFPX: Changed sensors spacing within all double-disks + Increased distance between Disks 6 and 7 + Put TBPX portcards between Disks 6 and 7.
* T18: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.6) TFPX: Shift all double-disks by + 25mm in Z. All services volumes (& TBPX portcards) are also shifted.

Calorimeters:
* C4: HGCal (v9) + Phase2 HCAL and EB
* C6: HGCal (v9) + HFNose + Phase2 HCAL and EB
* C8: HGCal (v10 post TDR HGCal Geometry) + Phase2 HCAL and EB + Tracker cables in calorimeter region
* C9: HGCal (v11 post TDR HGCal Geometry same as C8 + corner centering for HE part)
* C10: HGCal (as in C9) + HFNose with corrected wafer size + Phase2 HCAL and EB

Muon system:
* M2: Phase2 muon system for TDR w/ GE2/1, ME0, RE3/1, RE4/1 (incl. granularity in ME0, staggered GE2/1)
* M3: same as M2 with change to the number of iRPC strips from 192 to 96 as in TDR
* M4: same as M3 but no Overlaps inside the Muon System and MB4Shields completed

Fast Timing system:
* I5: Fast Timing detector (LYSO barrel, silicon endcap), full description with passive materials, LYSO bars along z flat
* I7: Fast Timing detector (LYSO barrel, silicon endcap), full description with passive materials, LYSO bars along phi flat
* I9: Same as I7 but with ETL in the position defined in O3
* I10: Same as I9 w/ material adjustments
* I11: Same as I10, xml reorganized, comparison base for new ETL and DD4hep migration
* I12: Starting from I11, new ETL layout from MTD TDR

The script also handles the common and forward elements of the geometry:
* O2: detailed cavern description
* O3: O2 + changes due to modified CALO region due to changes in the Endcap part
* F2: modifications needed to accommodate detailed cavern, ZDC description is removed.
* F3: same as F2 but changes due to HFNose
* O4: same of O3 but no Overlaps inside the Muon System 

Several detector combinations have been generated:
* D35 = T6+C4+M2+I5+O2+F2 
* D41 = T14+C8+M3+I9+O3+F2
* D43 = T14+C4+M3+I7+O2+F2
* D44 = T14+C6+M3+I7+O2+F3
* D45 = T15+C8+M3+I10+O3+F2
* D46 = T15+C9+M3+I10+O3+F2
* D47 = T15+C10+M3+I10+O3+F3
* D48 = T16+C9+M3+I10+O3+F2
* D49 = T15+C9+M4+I10+O4+F2
* D50 = T15+C9+M4+I11+O4+F2
* D51 = T17+C9+M4+I10+O4+F2
* D52 = T18+C9+M4+I10+O4+F2
* D53 = T15+C9+M4+I12+O4+F2

D35 is the MTD TDR baseline, D41 is the L1T TDR baseline, and D49 is the HLT TDR baseline.
