#ifndef DPGAnalysis_SiStripTools_SiStripQualityHistoryPlots_h
#define DPGAnalysis_SiStripTools_SiStripQualityHistoryPlots_h

class TH1D;
class TCanvas;
class TFile;

TH1D* AverageRunBadChannels(TFile& ff, const char* module, const char* histo, bool excludeLastBins=false);
TCanvas* StripCompletePlot(TFile& ff, const char* module, bool excludeLastBins=false);

#endif // DPGAnalysis_SiStripTools_SiStripQualityHistoryPlots_h
