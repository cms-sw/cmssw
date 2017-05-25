The Phase2 geometries are automatically created using the script [generate2023Geometry.py](./scripts/generate2023Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T1: Phase2 tilted tracker (v2016-04-12) w/ phase 1 (extended) pixel - DEPRECATED, superseded by T3
* T2: Phase2 flat tracker (v2016-04-12) w/ phase 1 (extended) pixel- DEPRECATED, superseded by T4
* T3: Phase2 tilted tracker (v3.6.5) w/ phase 2 pixel (v4.0.2.6) - DEPRECATED, superseded by T5
* T4: Phase2 flat tracker (v2016-04-12) w/ phase 2 pixel (v4.0.2.6)
* T5: Phase2 tilted tracker (v6.1.3) w/ phase 2 pixel (v4.0.2.5) 

Calorimeters:
* C1: Run2 calorimeters
* C2: HGCal (v7) + Phase2 HCAL and EB
* C3: HGCal (v8) + Phase2 HCAL and EB

Muon system:
* M1: Phase2 muon system (TP baseline) w/ GE2/1, ME0, RE3/1, RE4/1
* M2: Phase2 muon system for TDR (incl. granularity in ME0, staggered GE2/1)

Fast Timing system:
* I1: No Fast Timing detector
* I2: Fast Timing detector (LYSO barrel, silicon endcap)

The script also handles the common and forward elements of the geometry:
* O1: which is not expected to change
* O2: detailed cavern description
* F1: which is not expected to change
* F2: modifications needed to accommodate detailed cavern, ZDC description is removed.

Several detector combinations have been generated:
* D10 = T4+C1+M1+I1+O1+F1
* D11 = T5+C2+M1+I1+O1+F1 
* D14 = T5+C2+M2+I1+O2+F2 
* D16 = T5+C3+M2+I1+O1+F1
* D17 = T5+C3+M2+I1+O2+F2 
* D18 = T5+C2+M1+I2+O1+F1
Currently, D17 is considered to be the baseline for the Phase 2 Muon and Barrel TDRs.


