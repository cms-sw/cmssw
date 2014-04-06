class TFile;
class TCanvas;
class TH1D;
class TText;

void PlotOccupancyMap(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color);
void PlotOccupancyMapPhase2(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color);
void printFrame(TCanvas* c, TH1D* h, TText* t, const int frame, const int min, const int max);

