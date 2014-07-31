#ifndef __DQMOffline_PFTau_PFMuonDQMAnalyzer__
#define __DQMOffline_PFTau_PFMuonDQMAnalyzer__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DQMOffline/PFTau/interface/PFCandidateMonitor.h"


class PFMuonDQMAnalyzer: public edm::EDAnalyzer {
 public:
  
  PFMuonDQMAnalyzer(const edm::ParameterSet& parameterSet);
  
 private:
  void analyze(edm::Event const&, edm::EventSetup const&);
  void beginJob() ;
  void endJob();

  void storeBadEvents(edm::Event const&, float& val);

  //edm::EDGetTokenT< edm::View<reco::Candidate> > myCand_;
  //edm::EDGetTokenT< edm::View<reco::Candidate> > myMatchedCand_;
  edm::EDGetTokenT< edm::View<reco::Muon> > myCand_;
  edm::EDGetTokenT< edm::View<reco::Muon> > myMatchedCand_;
  //edm::EDGetTokenT< edm::View<reco::Muon> > myMuonMatchedCand_;
  edm::InputTag matchLabel_;
  edm::InputTag inputLabel_;
  std::string benchmarkLabel_;
  bool createEfficiencyHistos_;

  double ptBase_;
  double ptNotPF_;

  PFCandidateMonitor pfCandidateMonitor_;

  edm::ParameterSet pSet_;

  int nBadEvents_;
};

#endif 
