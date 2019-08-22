#ifndef CSCObjects_CSCGainsStudyHistograms_H
#define CSCObjects_CSCGainsStudyHistograms_H

/** \class CSCGainsStudyHistograms
 * Collection of histograms for plotting gain correction weights
 *  in each chambers used in MTCC.
 *
 * Author: D. Fortin  - UC Riverside
 */

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include <string>
#include <iostream>

class HCSCGains {
public:
  /// Constructor from collection name
  HCSCGains(std::string name_) {
    TString N = name_.c_str();
    name = N;
    hGains = new TH1F("hGain_" + name, name, 200, 0.75, 3.75);
    hGaindiff = new TH1F("hGaindiff_" + name, name, 101, -0.101, 0.101);
    hGainvsch = new TH2F("hGainvsch_" + name, name, 600, 100.5, 700.5, 81, 0.7975, 1.225);
  }

  /// Constructor from collection name and TFile.
  HCSCGains(TString name_, TFile *file) {
    name = name_;
    hGains = (TH1F *)file->Get("hGains_" + name);
    hGaindiff = (TH1F *)file->Get("hGaindiff_" + name);
    hGainvsch = (TH2F *)file->Get("hGainvsch_" + name);
  }

  /// Destructor
  virtual ~HCSCGains() {
    delete hGains;
    delete hGaindiff;
    delete hGainvsch;
  }

  // Operations

  /// Fill all the histos
  void Fill(float weight, float weight0, int channel, int layer) {
    float weightx = weight + float(layer - 1) / 2.;
    hGains->Fill(weightx);

    int id = channel + 100 * layer;
    hGainvsch->Fill(id, weight);
    if (weight0 > 0. && weight > 0.)
      hGaindiff->Fill(weight - weight0);
  }

  /// Write all the histos to currently opened file
  void Write() {
    hGains->GetXaxis()->SetTitle("weight + (layer - 1)/2");
    hGains->Write();

    hGaindiff->GetXaxis()->SetTitle("weight diff between 2 adjacent strips");
    hGaindiff->Write();

    hGainvsch->GetXaxis()->SetTitle("strip_id + 100 x layer");
    hGainvsch->GetYaxis()->SetTitle("weight");
    hGainvsch->Write();
  }

  TH1F *hGains;
  TH1F *hGaindiff;
  TH2F *hGainvsch;

  TString name;
};
#endif
