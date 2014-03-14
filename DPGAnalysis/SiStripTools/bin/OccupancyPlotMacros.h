#ifndef DPGAnalysis_SiStripTools_OccupancyPlotMacros_h
#define DPGAnalysis_SiStripTools_OccupancyPlotMacros_h

class TFile;

void PlotOccupancyMap(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color);
float combinedOccupancy(TFile* ff, const char* module, const int lowerbin, const int upperbin);

#endif // DPGAnalysis_SiStripTools_OccupancyPlotMacros_h

