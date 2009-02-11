#ifndef SIMJETCORRECTIONSANALYSIS_H
#define SIMJETCORRECTIONSANALYSIS_H
//---------------------------------------------------------------------------------   
//  The code to make Jet correction plots from the Simulated jets
//  
//  November 21, 2006   Anwar A Bhatti  The Rockefeller University, New York NY
//  
//----------------------------------------------------------

#include "TGraph.h"
#include "TGraphErrors.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TFile.h"
#include "TNamed.h"

#include <vector>
#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/JetReco/interface/CaloJetfwd.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"



class SimJetResponseAnalysis : public edm::EDAnalyzer {

 public:

  explicit SimJetResponseAnalysis(edm::ParameterSet const& cfg);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  virtual void endJob();

  SimJetResponseAnalysis();
  void analyze(const reco::GenJetCollection& genjets,const reco::CaloJetCollection& recjets,
               const reco::GenMETCollection& genmet, const reco::CaloMETCollection& recmet);
  void done();


  void fillHist1D(const TString& histName, const Double_t& x, const Double_t& wt=1.0);
  void fillHist2D(const TString& histName, const Double_t& x, const Double_t& y,  const Double_t& wt=1.0);


  void bookHistograms();
  void bookGeneralHistograms();

  void bookMetHists(const TString& prefix);
  template <typename T> void fillMetHists(const T& mets, const TString& prefx);

  void bookJetHistograms(const TString& prefix);
  template <typename T> void fillJetHists(const T& jets, const TString& prefx);

  int GetPtBin(double GenPtJet);
  int TowerNumber(double eta);
  int GetEtaBin(double eta);
  void GetSimJetResponse();
  void bookSimJetResponse();

  void SimulatedJetResponse(const reco::GenJetCollection& genjets,const reco::CaloJetCollection& calojets);

private:

  // input variables

  std::string histogramFile_;

  int NJetMax_;
  double MatchRadius_;
  double RecJetPtMin_;
  std::vector<double> GenJetPtBins_;
  std::vector<double> RecJetEtaBins_;
  int  NPtBins;
  int  NEtaBins;

  TFile* hist_file_; // pointer to Histogram file

  // use the map function to store the histograms

  std::map<TString, TH1*> m_HistNames1D;
  std::map<TString, TH2*> m_HistNames2D;
  std::map<TString, TProfile*> m_HistNamesProf;
  std::map<TString, TGraph*> m_HistNamesGraph;

  std::string genjets_,recjets_,genmet_,recmet_;
  TString gjetpfx, rjetpfx,gmetpfx, rmetpfx,calopfx;

};

#endif
