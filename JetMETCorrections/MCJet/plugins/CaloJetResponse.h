#ifndef CALOJETRESPONSE_H
#define CALOJETRESPONSE_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

namespace cms
{
class CaloJetResponse : public edm::EDAnalyzer 
{
  public:
     explicit CaloJetResponse(edm::ParameterSet const& cfg);
     virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
     virtual void endJob();
     CaloJetResponse();
     void analyze(const reco::GenJetCollection& genjets,const reco::CaloJetCollection& recjets);
     void done();
     void fillHist1D(const TString& histName, const Double_t& x);
     void fillHist2D(const TString& histName, const Double_t& x, const Double_t& y);
     void bookHistograms();  
     void CalculateJetResponse(const reco::GenJetCollection& genjets,const reco::CaloJetCollection& recjets);
     int GetBin(double x, std::vector<double> a);

  private:
     // input variables
     int    NJetMax_;
     int    NGenPtBins_;
     int    NEtaBins_; 
     double MatchRadius_;
     double RecJetPtMin_;
     std::vector<double> GenJetPtBins_;
     std::vector<double> EtaBoundaries_;
     std::string histogramFile_;
     std::string genjets_;
     std::string recjets_;
     std::map<TString, TH1*> m_HistNames1D;
     std::map<TString, TH2*> m_HistNames2D;
     TFile* hist_file_; 
};
}
#endif
