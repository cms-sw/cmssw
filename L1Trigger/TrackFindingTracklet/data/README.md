# L1Trigger-TrackFindingTracklet

This repository contains various data files that are needed to run the L1 tracking. 

*** WIRING ***

These contain the detailed wiring of the various modules for the "hourglass" phi sector definition, 
for either the baseline tracking (wires_hourglass.dat) or the extended tracking that includes displaced 
(triplet) seeding (wires_hourglassExtended.dat)

wires_hourglass.dat 

wires_hourglassExtended.dat 

*** PROCESSING MODULES *** 

These contain the detailed list of processing modules used for the baseline or extended tracking (e.g. VMRouter, TrackletEngine, TrackletCalculator, ...) 

processingmodules_hourglass.dat

processingmodules_hourglassExtended.dat

*** MEMORY MODULES *** 

These similarly contain the detailed memory modules (e.g. InputLink, AllStubs, FullMatch, TrackletParameters, ...)

memorymodules_hourglass.dat

memorymodules_hourglassExtended.dat

*** LUTs FOR EXTENDED TRACKING *** 

These contain LUTs needed for the extended tracking (that allows specifically reconstructing displaced trajectories). Lines in files correspond to different indices, based on phi coordinate and bend of stubs (for the TrackletEngineDisplaced (table_TED) these are the two stubs considered as candidate stub pair, while for the Triplet Engine (table_TRE) it is one of the stubs from the initial stub pair plus third stub forming candidate triplet). These are created through training on muon gun samples. 

table_TED/ => tables for TrackletEngineDisplaced.

table_TRE/ => tables for TripletEngine.

*** CHI2 FIT ***

This contains track derivatives used for the chi2-based track fitting. 

fitpattern.txt

*** MODULES & CABLING *** 

This contains information about the modules and the links that they are associated with from the DTC.

modules_T5v3_27SP_nonant_tracklet.dat

calcNumDTCLinks.txt

dtclinklayerdisk.dat

