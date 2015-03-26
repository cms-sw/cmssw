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
