#ifndef CSCObjects_CSCGainsStudyHistograms_H
#define CSCObjects_CSCGainsStudyHistograms_H

/** \class CSCGainsStudyHistograms
 * Collection of histograms for plotting gain correction weights
 *  in each chambers used in MTCC.
 *
 * Author: D. Fortin  - UC Riverside
 */

#include "TH1F.h"
#include "TFile.h"
#include "TString.h"
#include <string>
#include <iostream>


class HCSCGains {
  
 public:
  /// Constructor from collection name
  HCSCGains(std::string name_) {
    TString N = name_.c_str();
    name=N;
    hGains = new TH1F("hGain_"+name, name, 101, 0.745, 1.255);
    hGaindiff = new TH1F("hGaindiff_"+name, name, 101, -0.255, 0.255);

  }
  
  /// Constructor from collection name and TFile.
  HCSCGains(TString name_, TFile* file) {
    name=name_;
    hGains     = (TH1F *) file->Get("hGains_"+name);
    hGaindiff  = (TH1F *) file->Get("hGaindiff_"+name);
  }
  
  /// Destructor
  virtual ~HCSCGains() {
    delete  hGains;
    delete  hGaindiff;
  }

  
  // Operations

  /// Fill all the histos
  void Fill(float weight, float weight0) {
    hGains->Fill(weight);
    if (weight0 > 0. && weight > 0. ) hGaindiff->Fill(weight-weight0);
  }


  /// Write all the histos to currently opened file
  void Write() {
    hGains->GetXaxis()->SetTitle("Gain correction weight");
    hGains->Write();
    hGaindiff->GetXaxis()->SetTitle("weight difference between 2 adjacent strips");
    hGaindiff->Write();
  }


  TH1F  *hGains;
  TH1F  *hGaindiff;  
  
  TString name;

};
#endif


