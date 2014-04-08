#ifndef DPGAnalysis_SiStripTools_SiStripQualityHistoryPlots_h
#define DPGAnalysis_SiStripTools_SiStripQualityHistoryPlots_h

class TH1D;
class TCanvas;
class TFile;

TH1D* AverageRunBadChannels(TFile& ff, const char* module, const char* histo, const bool excludeLastBins=false);
TCanvas* StripCompletePlot(TFile& ff, const char* module, const bool excludeLastBins=false);

#endif // DPGAnalysis_SiStripTools_SiStripQualityHistoryPlots_h
