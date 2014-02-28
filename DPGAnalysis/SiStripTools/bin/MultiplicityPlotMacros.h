#ifndef DPGAnalysis_SiStripTools_MultiplicityPlotMacros_h
#define DPGAnalysis_SiStripTools_MultiplicityPlotMacros_h

class TFile;
class TH1D;

void PlotPixelMultVtxPos(TFile* ff, const char* module);
TH1D* AverageRunMultiplicity(TFile& ff, const char* module, const bool excludeLastBins, const char* histo);

#endif // DPGAnalysis_SiStripTools_MultiplicityPlotMacros_h
