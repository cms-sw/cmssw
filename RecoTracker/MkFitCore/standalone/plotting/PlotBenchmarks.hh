#ifndef _PlotBenchmarks_
#define _PlotBenchmarks_

#include "Common.hh"

typedef std::vector<TGraphErrors*> TGEVec;

class PlotBenchmarks {
public:
  PlotBenchmarks(const TString& arch, const TString& sample, const TString& suite);
  ~PlotBenchmarks();
  void RunBenchmarkPlots();
  void MakeOverlay(const TString& text,
                   const TString& title,
                   const TString& xtitle,
                   const TString& ytitle,
                   const Double_t xmin,
                   const Double_t xmax,
                   const Double_t ymin,
                   const Double_t ymax);
  void GetGraphs(
      TGEVec& graphs, const TString& text, const TString& title, const TString& xtitle, const TString& ytitle);

private:
  const TString arch;
  const TString sample;
  const TString suite;

  TFile* file;
};

#endif
