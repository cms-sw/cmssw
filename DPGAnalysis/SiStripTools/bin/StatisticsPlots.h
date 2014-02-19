#ifndef DPGAnalysis_SiStripTools_StatisticsPlots_h
#define DPGAnalysis_SiStripTools_StatisticsPlots_h

#include <vector>

class TH1F;
class TFile;
class TH1D;
class TGraphAsymmErrors;
class TH2F;

void DeadTimeAPVCycle(TH1F* hist, const std::vector<int>& bins);
TH1F* CombinedHisto(TFile& ff, const char* module, const char* histname);
TH1F* TimeRatio(TFile& ff, const char* modulen, const char* moduled, const int irun, const int rebin=1);
TH1D* SummaryHisto(TFile& ff, const char* module);
TH1D* SummaryHistoRatio(TFile& f1, const char* mod1, TFile& f2, const char* mod2, const char* hname);
TGraphAsymmErrors* SummaryHistoRatioGraph(TFile& f1, const char* mod1, TFile& f2, const char* mod2, const char* hname);
TH2F* Combined2DHisto(TFile& ff, const char* module, const char* histname);
void StatisticsPlots(const char* fullname, const char* module, const char* label, const char* postfix, const char* shortname,
		     const char* outtrunk);

#endif //  DPGAnalysis_SiStripTools_StatisticsPlots_h
