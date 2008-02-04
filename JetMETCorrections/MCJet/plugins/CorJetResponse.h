#ifndef CORJETRESPONSE_H
#define CORJETRESPONSE_H

#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
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
class CorJetResponse : public edm::EDAnalyzer 
{
  public:

    explicit CorJetResponse(edm::ParameterSet const& cfg);
    virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
    virtual void endJob();

    CorJetResponse();
    void analyze(const reco::GenJetCollection& genjets,const reco::CaloJetCollection& recjets);
    void done();
    void FillHist1D(const TString& histName, const Double_t& x);
    void FillHist2D(const TString& histName, const Double_t& x, const Double_t& y);
    void BookHistograms();
    void BookJetResponse();
    void BookJetPt();
    void CalculateJetResponse(const reco::GenJetCollection& genjets,const reco::CaloJetCollection& recjets);
    int  GetBin(double x, std::vector<double> boundaries);

  private:
    int    NPtBins_;
    int    NEtaBins_;
    int    NJetMax_;
    double MatchRadius_;
    double RecJetPtMin_;
    std::string histogramFile_;
    std::string genjets_;
    std::string recjets_;
    std::vector<double> JetPtBins_;
    std::vector<double> EtaBoundaries_;
    TFile* hist_file_; // pointer to Histogram file
    // use the map function to store the histograms
    std::map<TString, TH1*> m_HistNames1D;
    std::map<TString, TH2*> m_HistNames2D;
};
}
#endif
