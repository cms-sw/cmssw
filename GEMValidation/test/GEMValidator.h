#ifndef GEMValidation_GEMValidator_h
#define GEMValidation_GEMValidator_h

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TH1D.h"
#include "TPad.h"
#include "TStyle.h"
#include "TString.h"
#include "TAxis.h"
#include "TArrayD.h"
#include "TPDF.h"
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string>
#include <sstream>


class GEMValidator
{
 public:
  enum Selection{Muon, NonMuon, All};

  GEMValidator();
  ~GEMValidator();
  
  void produceSimHitValidationPlots(const Selection& key); 
  void produceDigiValidationPlots();
  void produceGEMCSCPadDigiValidationPlots(const std::string treeName);
  void produceTrackValidationPlots();
  
  void setEtaBinLabels(const TH1D* h);
  template<typename T> const std::string to_string(T const& value); 
  
 private:
  TString fileExtension_;
  TString simHitFileName_;
  TString digiFileName_;
};

#endif
