The Phase2 geometries are automatically created using the script [generate2023Geometry.py](./scripts/generate2023Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T4: Phase2 flat tracker (v2016-04-12) w/ phase 2 pixel (v4.0.2.6)
* T5: Phase2 tilted tracker (v6.1.3) w/ phase 2 pixel (v4.0.2.5) 

Calorimeters:
* C3: HGCal (v8) + Phase2 HCAL and EB

Muon system:
* M2: Phase2 muon system for TDR w/ GE2/1, ME0, RE3/1, RE4/1 (incl. granularity in ME0, staggered GE2/1)

Fast Timing system:
* I1: No Fast Timing detector
* I2: Fast Timing detector (LYSO barrel, silicon endcap)

The script also handles the common and forward elements of the geometry:
* O2: detailed cavern description
* F2: modifications needed to accommodate detailed cavern, ZDC description is removed.

Several detector combinations have been generated:
* D17 = T5+C3+M2+I1+O2+F2 
* D19 = T5+C3+M2+I2+O2+F2 
* D20 = T4+C3+M2+I1+O2+F2 

Currently, D17 is considered to be the baseline for the Phase 2 Muon and Barrel TDRs.


