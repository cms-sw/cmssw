#ifndef RecoLocalMuon_DTRecHitHistograms_H
#define RecoLocalMuon_DTRecHitHistograms_H

/** \class DTRecHitHistograms
 *  Collection of histograms for RecHits.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */


#include "TH2.h"
#include "TString.h"
#include <string>
#include <iostream>

class H1DRecHit {
public:
  /// Constructor
  H1DRecHit(std::string name_) {
    TString N = name_.c_str();
    name=N;
    
    hRecDist = new TH1F(N+"_hRecDist", "1D DTRecHit distance from wire (cm)", 100, 0, 2.5);
    hSimDist = new TH1F(N+"_hSimDist", "Mu SimHit distance from wire (cm)", 100, 0, 2.5);
    hResDist = new TH1F(N+"_hResDist", "1D DTRecHit residual on the distance from wire (cm)",
			100, -0.5, 0.5);
    hResDistVsDist = new TH2F(N+"_hResDistVsDist",
			      "1D DTRecHit residual on the distance from wire vs distance (cm)",
			      100, 0, 2.5, 100, -0.5, 0.5);
  }


  H1DRecHit(TString name_, TFile* file) {
    name=name_;
    hRecDist          = (TH1F *) file->Get(name+"_hRecDist");
    hSimDist          = (TH1F *) file->Get(name+"_hSimDist");
    hResDist          = (TH1F *) file->Get(name+"_hResDist");
    hResDistVsDist    = (TH2F *) file->Get(name+"_hResDistVsDist");
  }
  /// Destructor
  virtual ~H1DRecHit() {
    delete hRecDist;
    delete hSimDist;
    delete hResDist;
    delete hResDistVsDist;
  }

  // Operations
  void Fill(float recDist, float simDist) {
    hRecDist->Fill(recDist);
    hSimDist->Fill(simDist);
    hResDist->Fill(recDist-simDist);
    hResDistVsDist->Fill(simDist, recDist-simDist);
  }

  void Write() {
    hRecDist->Write();
    hSimDist->Write();
    hResDist->Write();
    hResDistVsDist->Write();
  }
  
  TH1F *hRecDist;
  TH1F *hSimDist;
  TH1F *hResDist;
  TH2F *hResDistVsDist;
  TString name;
};
#endif


