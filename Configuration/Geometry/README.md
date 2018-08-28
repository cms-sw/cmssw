The Phase2 geometries are automatically created using the script [generate2023Geometry.py](./scripts/generate2023Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T5: Phase2 tilted tracker (v6.1.3) w/ phase 2 pixel (v4.0.2.5) 
* T6: Phase2 tilted tracker (v6.1.4) w/ phase 2 pixel (v4.0.4) (TEDD slighly rotated + Inner Tracker barrel has lower radii than TDR T5 geometry)
* T7: Phase2 tilted tracker (v6.1.4) w/ phase 2 pixel (v4.2.0) (same as T6 but 25x100 in 1x2 modules, 50x200 in 2x2 modules)
* T8: Phase2 tilted tracker (v6.1.4) w/ phase 2 pixel (v4.2.1) (same as T6 but 25x100 in 1x2 modules, 100x100 in 2x2 modules)
* T9: Same as T5, but with an increase of 20% of the material (stress test studies)
* T10: Same as T5 but with an increase of 50% of the material (stress test studies) 
* T11: Phase2 tilted tracker (v6.1.4) w/ phase 2 pixel (v4.0.5) (same as T6 but 50x50 in all modules)

Calorimeters:
* C3: HGCal (v8) + Phase2 HCAL and EB
* C4: HGCal (v9) + Phase2 HCAL and EB
* C5: HGCal (v9 without virtual wafers) + Phase2 HCAL and EB
* C6: HGCal (v9) + HFNose + Phase2 HCAL and EB

Muon system:
* M2: Phase2 muon system for TDR w/ GE2/1, ME0, RE3/1, RE4/1 (incl. granularity in ME0, staggered GE2/1)

Fast Timing system:
* I1: No Fast Timing detector
* I2: Fast Timing detector (LYSO barrel, silicon endcap), only sensitive layers
* I3: Fast Timing detector (LYSO barrel, silicon endcap), full description with passive materials, LYSO tiles
* I4: Fast Timing detector (LYSO barrel, silicon endcap), full description with passive materials, LYSO bars

The script also handles the common and forward elements of the geometry:
* O2: detailed cavern description
* F2: modifications needed to accommodate detailed cavern, ZDC description is removed.
* F3: same as F2 but changes due to HFNose

Several detector combinations have been generated:
* D17 = T5+C3+M2+I1+O2+F2 
* D19 = T5+C3+M2+I2+O2+F2 
* D21 = T6+C3+M2+I1+O2+F2 
* D22 = T7+C3+M2+I1+O2+F2 
* D23 = T8+C3+M2+I1+O2+F2 
* D24 = T6+C3+M2+I3+O2+F2 
* D25 = T6+C3+M2+I4+O2+F2 
* D26 = T9+C3+M2+I1+C2+F2
* D27 = T10+C3+M2+I1+O2+F2
* D28 = T6+C4+M2+I1+O2+F2
* D29 = T11+C3+M2+I1+O2+F2 
* D30 = T6+C5+M2+I1+O2+F2
* D31 = T6+C6+M2+I1+O2+F3

Currently, D17 is considered to be the baseline for the Phase 2 Muon and Barrel TDRs.
