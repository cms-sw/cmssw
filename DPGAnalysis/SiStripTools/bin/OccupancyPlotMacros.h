#ifndef DPGAnalysis_SiStripTools_OccupancyPlotMacros_h
#define DPGAnalysis_SiStripTools_OccupancyPlotMacros_h

#include <vector>
#include <string>
#include <utility>

class TFile;
class TCanvas;
class TH1D;
class TH1;
class TProfile;
class TText;

struct SubDetParams {
  std::string label;
  int min;
  int max;
};

float linear(float x);
float logarithm(float x);
std::pair<float,float> phase2bin(int i);
void PlotOccupancyMap(TFile* ff, const char* module, float min, float max, float mmin, float mmax, int color);
void PlotOccupancyMapPhase1(TFile* ff, const char* module, float min, float max, float mmin, float mmax, int color);
void PlotOccupancyMapPhase2(TFile* ff, const char* module, float min, float max, float mmin, float mmax, int color);
void PlotOccupancyMapGeneric(TFile* ff, const char* module, float min, float max, float mmin, float mmax, int color,
			     std::pair<float,float>(*size)(int), const std::vector<SubDetParams>& vsub);
void printFrame(TCanvas* c, TH1D* h, const char* label, int frame, int min, int max, bool same=false);
float combinedOccupancy(TFile* ff, const char* module, int lowerbin, int upperbin);
void PlotOnTrackOccupancy(TFile* ff, const char* module, const char* ontrkmod, float mmin, float mmax, int color);
void PlotOnTrackOccupancyPhase1(TFile* ff, const char* module, const char* ontrkmod, float mmin, float mmax, int color);
void PlotOnTrackOccupancyPhase2(TFile* ff, const char* module, const char* ontrkmod, float mmin, float mmax, int color);
void PlotOnTrackOccupancyGeneric(TFile* ff, const char* module, const char* ontrkmod, float mmin, float mmax, int color,
				 std::pair<float,float>(*size)(int), const std::vector<SubDetParams>& vsub);
void PlotDebugFPIX_XYMap(TFile* ff, const char* module, unsigned int offset, const char* name);
void PlotTrackerXsect(TFile* ff, const char* module);
TCanvas* drawMap(const char* cname, const TH1* hval, const TProfile* averadius, const TProfile* avez,float mmin, float mmax, 
		 std::pair<float,float>(*size)(int), float(*scale)(float), int color, const char* ptitle="");
TH1D* TrendPlotSingleBin(TFile* ff, const char* module, const char* hname, int bin);

#endif // DPGAnalysis_SiStripTools_OccupancyPlotMacros_h

