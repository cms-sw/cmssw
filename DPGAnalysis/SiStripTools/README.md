## Plugins
### APVCyclePhaseDebuggerFromL1TS
It is an `EDAnalyzer` which can be used to monitor the content of the `Level1TriggerScalersCollection` produced by SCAL. Histograms are filled for each event with the orbit number of the last: resynch, hard reset, EC0, OC0, Test Enable and Start signals. Therefore the histograms show spikes at the orbit number values when the signals have been issued with the a number of entries equal to the number of events between two signals. In addition two histograms are filled with the differences of the orbit numbers between EC0 and resync and resync and hard reset every time a new resync or a new hard reset has been issued. The configuration requires simply the name of the product with the `Level1TriggerScalersCollection` and, usually, it is `scalersRawToDigi`, as defined in the default configuration.

### APVCyclePhaseProducerFromL1TS
It is an `EDProducer` which produces an `APVCyclePhaseCollection` which contains, for each Tracker partition, an offset, between 0 and 69, to be used to compute the position of each event in the Tracker APV readout cycle from the orbit number and the bunch crossing number using `orbit*3564+bx-offset`. More details can be found [in this page](https://twiki.cern.ch/twiki/bin/view/CMS/APVReadoutCycle). The standard way of using it is to let the EventSetup provide the parameters using the record `SiStripConfObject` with the label `apvphaseoffsets` (configurable). If the configuration parameter `ignoreDB` is set to true, then the parameters are read from the configuration file and are: `defaultPartitionNames` usually equal to `TI,TO,TP,TM`, `defaultPhases` which contains a vector (one value per partition) of offsets which are used as offset when no resync has been issued, `magicOffset` which is used to correct the orbit number of the last resync to compute the phase when a resync has been issued using the expression: `(defaultPhase + (lastresyncorbit+magicOffset)*3564%70)%70`, and `useEC0` if we want to use `lastEC0orbit` instead of `lastresyncorbit` to compute the phase. Examples of configurations can be found in the `python` directory.

## ROOT Macros
This package contains a library of ROOT macros that can be loaded by executing `gSystem->Load("$CMSSW_BASE/lib/$SCRAM_ARCH/libDPGAnalysisSiStripToolsMacros.so")`
in the root interactive section

### PlotOccupancyMap
`PlotOccupancyMap(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color)`

It requires as input a pointer to a root file, `ff`, which contains a `TDirectory` named `module`, produced by the 
`OccupancyPlots` EDAnalyzer (in this package). It produces histograms and color map of the cluster occupancy (number
of strips or pixels divided by the number of channels), multiplicity (number of clusters divided by the number of 
channels) in each subset of modules and the histograms of the ratio between the occupancy and the multiplicity which can
be interpreted as the cluster size. Each bin in the histograms represents a subset of modules represented in the
Tracker cross-section map. The parameters `min` and `max` are used to set the scale of the occupancy map while `mmin` and
`mmax` are ysed to se the scale of the multiplicity map. The parameter `color` is used to define the color palette
(color = 0 is the usual rainbow palette). If the macro is executed twice without deleting the previous TCanvas, the 
plots of the second execution are superimposed to the previous ones in the TCanvas with the six windows. This macro
depends on the setting of the file `DPGAnalysis/SiStripTools/python/occupancyplotsselection_simplified_cff`.

### PlotOccupancyMapPhase1
As the macro `PlotOccupancyMap` but for results obtained with the phase 1 geometry

### PlotOccupancyMapPhase2
As the macro `PlotOccupancyMap` but for results obtained with the phase 2 (Technical Proposal) geometry

### PlotOnTrackOccupancy
`PlotOnTrackOccupancy(TFile* ff, const char* module, const char* ontrkmod, const float mmin, const float mmax, const int color)`

Similarly to the macro `PlotOccupancyMap` it requires as input a root file with the histograms produced by `OccupancyPlots`
but instead of one set of histograms it requires two of them in the TDirectory names `module` and `ontrkmod`. The macro
produces histograms and a Tracker cross-section map of the ratio of the average cluster multiplicity in the two set of
histograms. The purpose is to display the fraction of on-track clusters, for example.

### PlotOnTrackOccupancyPhase1
As the macro `PlotOnTrackOccupancy` but for results obtained with the phase 1 geometry

### PlotOnTrackOccupancyPhase2
As the macro `PlotOnTrackOccupancy` but for results obtained with the phase 2 geometry

## Scripts
### merge_and_move_generic_fromCrab.sh
Script which helps to merge root files produced by a crab2 job and saved in the `res` sudirectory of output crab directory. 
### merge_and_move_generic_fromCrab3.sh
Script which helps to merge root files produced by a crab3 job and saved in the `results` sudirectory of output crab directory. 


