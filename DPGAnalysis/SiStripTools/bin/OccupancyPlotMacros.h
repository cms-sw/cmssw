#ifndef DPGAnalysis_SiStripTools_OccupancyPlotMacros_h
#define DPGAnalysis_SiStripTools_OccupancyPlotMacros_h

class TFile;
class TCanvas;
class TH1D;
class TText;

void PlotOccupancyMap(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color);
void PlotOccupancyMapPhase2(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color);
void printFrame(TCanvas* c, TH1D* h, TText* t, const int frame, const int min, const int max);
float combinedOccupancy(TFile* ff, const char* module, const int lowerbin, const int upperbin);
void PlotOnTrackOccupancyPhase2(TFile* ff, const char* module, const char* ontrkmod, const float mmin, const float mmax, const int color);

#endif // DPGAnalysis_SiStripTools_OccupancyPlotMacros_h

#endif // DPGAnalysis_SiStripTools_OccupancyPlotMacros_h

