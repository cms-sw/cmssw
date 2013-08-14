#ifndef RecoLocalMuon_DTRecHitHistograms_H
#define RecoLocalMuon_DTRecHitHistograms_H

/** \class DTRecHitHistograms
 *  Collection of histograms for 1D DT RecHit test.
 *
 *  $Date: 2006/03/24 11:09:57 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */


#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include <string>



class H1DRecHit {
public:
  /// Constructor from collection name
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

  /// Constructor from collection name and TFile.
  /// It retrieves all the histos of the set from the file.
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
  /// Fill all the histos
  void Fill(float recDist, float simDist) {
    hRecDist->Fill(recDist);
    hSimDist->Fill(simDist);
    hResDist->Fill(recDist-simDist);
    hResDistVsDist->Fill(simDist, recDist-simDist);
  }

  /// Write all the histos to currently opened file
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


