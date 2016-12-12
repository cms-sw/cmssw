The Phase2 geometries are automatically created using the script [generate2023Geometry.py](./scripts/generate2023Geometry.py).

Different versions of various subdetectors can be combined. The available versions are:

Tracker:
* T1: Phase2 tilted tracker (v2016-04-12) w/ phase 1 (extended) pixel : DEPRECATED superseeded by T3
* T2: Phase2 flat tracker (v2016-04-12) w/ phase 1 (extended) pixel
* T3: Phase2 tilted tracker (v3.6.5) w/ phase 2 pixel (v4.0.2.6)

Calorimeters:
* C1: Run2 calorimeters
* C2: HGCal (v7) + Phase2 HCAL and EB
* C3: HGCal (v8) + Phase2 HCAL and EB

Muon system:
* M1: Phase2 muon system (TP baseline) w/ GE21, ME0, RE3/1, RE4/1
* M2: Phase2 muon system for TDR (incl. granularity in ME0)

Fast Timing system:
* I1: No Fast Timing detector
* I2: Fast Timing detector

The script also handles the common and forward elements of the geometry, which are not expected to change.

Several detector combinations have been generated:
* D1 = T3+C1+M1+I1
* D2 = T2+C1+M1+I1
* D3 = T3+C2+M1+I1
* D4 = T3+C2+M1+I1
* D5 = T3+C2+M1+I2
* D6 = T3+C1+M2+I1

With T1 being replaced by T3, D3 is now duplicated with D4, D3 is going to be removed at the next iteration 
Currently, D1 is considered to be the baseline for development purposes.


