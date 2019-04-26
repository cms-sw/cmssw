The Phase2 geometries are automatically created using the script [generate2023Geometry.py](./scripts/generate2023Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T5: Phase2 tilted tracker (v6.1.3) w/ phase 2 pixel (v4.0.2.5) 
* T6: Phase2 tilted tracker (v6.1.4) w/ phase 2 pixel (v4.0.4) (TEDD slighly rotated + Inner Tracker barrel has lower radii than TDR T5 geometry)
* T11: Phase2 tilted tracker (v6.1.4) w/ phase 2 pixel (v4.0.5) (same as T6 but 50x50 in all modules)
* T14: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.3) (Based from T12. OT: reduced envelope. IT: new chip size, different radii, 2x2 modules everywhere in TEPX, new ring paradigm in TEPX)
* T15: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.3) (Active geometry: same as T14. Material Budget: major update in IT, gathering info from recent Mechanical designs.)

Calorimeters:
* C3: HGCal (v8) + Phase2 HCAL and EB
* C4: HGCal (v9) + Phase2 HCAL and EB
* C5: HGCal (v9 without virtual wafers) + Phase2 HCAL and EB
* C6: HGCal (v9) + HFNose + Phase2 HCAL and EB
* C7: HGCal (v9a with inner support structure) + Phase2 HCAL and EB
* C8: HGCal (v10 post TDR HGCal Geometry) + Phase2 HCAL and EB + Tracker cables in calorimeter region

Muon system:
* M2: Phase2 muon system for TDR w/ GE2/1, ME0, RE3/1, RE4/1 (incl. granularity in ME0, staggered GE2/1)
* M3: same as M2 with change to the number of iRPC strips from 192 to 96 as in TDR

Fast Timing system:
* I1: No Fast Timing detector
* I2: Fast Timing detector (LYSO barrel, silicon endcap), only sensitive layers
* I3: Fast Timing detector (LYSO barrel, silicon endcap), full description with passive materials, LYSO tiles
* I4: Fast Timing detector (LYSO barrel, silicon endcap), full description with passive materials, LYSO bars
* I5: Fast Timing detector (LYSO barrel, silicon endcap), full description with passive materials, LYSO bars along z flat
* I6: Fast Timing detector (LYSO barrel, silicon endcap), full description with passive materials, LYSO bars along z flat no hole between modules
* I7: Fast Timing detector (LYSO barrel, silicon endcap), full description with passive materials, LYSO bars along phi flat
* I8: Fast Timing detector (LYSO barrel, silicon endcap), full description with passive materials, LYSO bars along phi flat, crystal thickness as I5
* I9: Same as I7 but with ETL in the position defined in O3

The script also handles the common and forward elements of the geometry:
* O2: detailed cavern description
* O3: O2 + changes due to modified CALO region due to changes in the Endcap part
* F2: modifications needed to accommodate detailed cavern, ZDC description is removed.
* F3: same as F2 but changes due to HFNose

Several detector combinations have been generated:
* D17 = T5+C3+M2+I1+O2+F2 
* D19 = T5+C3+M2+I2+O2+F2 
* D21 = T6+C3+M2+I1+O2+F2
* D24 = T6+C3+M2+I3+O2+F2 
* D25 = T6+C3+M2+I4+O2+F2 
* D28 = T6+C4+M2+I1+O2+F2
* D29 = T11+C3+M2+I1+O2+F2 
* D30 = T6+C5+M2+I1+O2+F2
* D31 = T6+C6+M2+I1+O2+F3
* D32 = T6+C7+M2+I1+O2+F2
* D33 = T6+C3+M2+I5+O2+F2 
* D34 = T6+C3+M2+I6+O2+F2 
* D35 = T6+C4+M2+I5+O2+F2 
* D38 = T6+C4+M2+I7+O2+F2
* D39 = T6+C4+M2+I8+O2+F2
* D40 = T14+C3+M2+I1+O2+F2
* D41 = T14+C8+M3+I9+O3+F2
* D42 = T15+C3+M2+I1+O2+F2

Currently, D17 is considered to be the baseline for the Phase 2 Muon and Barrel TDRs.
