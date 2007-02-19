#ifndef PLOTALLDISPLAY_H
#define PLOTALLDISPLAY_H

#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include <iostream>
#include "HistoManager.h"
#include "MyHcalClasses.h"
#include "TCanvas.h"
#include "TProfile.h"

class PlotAllDisplay {
public:
  PlotAllDisplay(const char *outfn) : m_f(outfn), histKeys(&m_f)
  {
    m_movie=0;
    n_movie=0;
  }
  void displaySummary(int ieta=0, int iphi=0, int evtType=4, int flavType=0);
  void displayOne(int ieta, int iphi, int depth, int evtType, int flavType);
  void displaySelector(int evtType, int flavType);
private:
  struct DisplaySetupStruct {
    std::string eventTypeStr;
    std::string flavTypeStr;
    int ieta, iphi;
  };
  std::vector<MyHcalDetId> spatialFilter(int ieta, int iphi,
					 const std::vector<MyHcalDetId>& inputs);
  TH1* bookMasterHistogram(DisplaySetupStruct& ss,
			   const std::string& basename,
			   int lo, int hi);
  MyHcalSubdetector getSubDetector(int ieta,int depth);
  TFile m_f;
  HistoManager histKeys;
  TCanvas* m_movie;
  int n_movie;
};

#endif
