// -*- C++ -*-
//
// Package:    L1TkBsCandidateAnalyzer
// Class:      L1TkBsCandidateAnalyzer
// 
/**\class L1TkBsCandidateAnalyzer L1Trigger/L1TTrackMatch/test/L1TkBsCandidateAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Authors:  R. Bhattacharya, S. Dutta and S. Sarkar
// Created:  Fri May 18 16:00:00 CET 2019
// $Id$
//

// system include files
#include <memory>
#include <vector>
#include <map>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

// L1Candidate etc.
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPhiCandidate.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPhiCandidateFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkBsCandidate.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkBsCandidateFwd.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "TH1F.h"

using namespace l1t;

class L1TkBsCandidateAnalyzer: public edm::EDAnalyzer {
public:

  using L1TTTrackType = TTTrack<Ref_Phase2TrackerDigi_>;
  using L1TTTrackCollectionType = std::vector<L1TTTrackType>;

  explicit L1TkBsCandidateAnalyzer(const edm::ParameterSet&);
  ~L1TkBsCandidateAnalyzer() {}
  
private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  static double PoissonError(double k, double N);
  static double BinomailError(double k, double N);
  void estimateRate(const TH1D* h, const std::string& label);

  struct HistoList {
    TH1F* nBs;
    TH1F* bsmass;
    TH1D* cutFlow;
  };
  std::map<std::string, HistoList> histoMap_;
  const edm::EDGetTokenT<L1TkBsCandidateCollection> tkBsToken_;
  const edm::EDGetTokenT<L1TkBsCandidateCollection> tkBsLooseWPToken_;
  const edm::EDGetTokenT<L1TkBsCandidateCollection> tkBsTightWPToken_;
};

L1TkBsCandidateAnalyzer::L1TkBsCandidateAnalyzer(const edm::ParameterSet& iConfig):
  tkBsToken_(consumes<L1TkBsCandidateCollection>(iConfig.getParameter<edm::InputTag>("L1TkBsCandidateInputTag"))),
  tkBsLooseWPToken_(consumes<L1TkBsCandidateCollection>(iConfig.getParameter<edm::InputTag>("L1TkBsCandidateLooseWPInputTag"))),
  tkBsTightWPToken_(consumes<L1TkBsCandidateCollection>(iConfig.getParameter<edm::InputTag>("L1TkBsCandidateTightWPInputTag")))
{
}
void L1TkBsCandidateAnalyzer::beginJob() {
  edm::Service<TFileService> fs;

  std::vector<std::string> wpList {"Medium", "Loose", "Tight"};
  for (auto const& v: wpList) {
    HistoList l;
    string hn = "nBs" + v + "WP";
    string ht = "No. of Bs Candidates for " + v + "WP"; 
    l.nBs     = fs->make<TH1F>(hn.c_str(), ht.c_str(), 4, -0.5, 3.5);

    hn = "bsmass" + v + "WP";
    ht = "Bs Candidate Mass for " + v + "WP"; 
    l.bsmass  = fs->make<TH1F>(hn.c_str(), ht.c_str(), 100, 5, 6);

    hn = "cutFlow" + v + "WP";
    ht = "Event Selection Cut flow for " + v + "WP"; 
    l.cutFlow = fs->make<TH1D>(hn.c_str(), ht.c_str(), 3, -0.5, 2.5);

    histoMap_.insert({v, l});
  }
}
void L1TkBsCandidateAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::map<std::string, const edm::EDGetTokenT<L1TkBsCandidateCollection>> tokenMap {
      {"Medium", tkBsToken_},
      {"Loose",  tkBsLooseWPToken_}, 
      {"Tight",  tkBsTightWPToken_}
  }; 
  for (auto el: tokenMap) {
    if (histoMap_.find(el.first) != histoMap_.end()) {
      const HistoList& hl = histoMap_.find(el.first)->second;
      hl.cutFlow->Fill(0);
      // the L1TkBsCandidate
      edm::Handle<L1TkBsCandidateCollection> collHandle;
      bool res = iEvent.getByToken(el.second, collHandle);
      if (res && collHandle.isValid()) {
        hl.cutFlow->Fill(1);
        L1TkBsCandidateCollection bsColl = *(collHandle.product());
        hl.nBs->Fill(bsColl.size());
        if (bsColl.size() > 0) hl.cutFlow->Fill(2);
        for (auto const& v: bsColl) hl.bsmass->Fill(v.mass());
      }
      else {
        std::cerr << "analyze: L1TkBsCandidateCollection for InputTag L1TkBsCandidateInputTag not found!" << std::endl; 
      }
    }
  }
}
void L1TkBsCandidateAnalyzer::endJob() {
  for (auto const& el: histoMap_)
    estimateRate(el.second.cutFlow, el.first);
}
void L1TkBsCandidateAnalyzer::estimateRate(const TH1D* h, const std::string& label) {
  constexpr double f_rate = 30000; // 30MHz, used for rate calculation
  double k = h->GetBinContent(3);
  double N = h->GetBinContent(1);
  double rate = (k/N) * f_rate;
  double err_Poisson  = L1TkBsCandidateAnalyzer::PoissonError(k,N) * f_rate;
  double err_Binomial = L1TkBsCandidateAnalyzer::BinomailError(k,N) * f_rate;
  std::cout << "BsToPhiPhiTo4K Rate @L1 [kHz] for " << label << "WP" << std::endl; 
  std::cout << std::setprecision(3);
  std::cout << std::setw(10) << "Rate" 
	    << std::setw(14) << "Poisson Error" 
	    << std::setw(15) << "Binomial Error" 
	    << std::endl;
  std::cout << std::setw(10) << rate
	    << std::setw(14) << err_Poisson 
            << std::setw(15) << err_Binomial
	    << std::endl;
}
double L1TkBsCandidateAnalyzer::PoissonError(double k, double N) {
  return std::sqrt(k * (N+k) / std::pow(N,3));
}
double L1TkBsCandidateAnalyzer::BinomailError(double k, double N) {
  return (1./N) * std::sqrt(k * (1 - k/N));
} 
// define this as a plug-in
DEFINE_FWK_MODULE(L1TkBsCandidateAnalyzer);
