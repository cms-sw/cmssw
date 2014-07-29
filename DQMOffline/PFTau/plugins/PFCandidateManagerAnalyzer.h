#ifndef __DQMOffline_PFTau_PFCandidateManagerAnalyzer__
#define __DQMOffline_PFTau_PFCandidateManagerAnalyzer__


#include "DQMOffline/PFTau/plugins/BenchmarkAnalyzer.h"
#include "DQMOffline/PFTau/interface/PFCandidateManager.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

class TH1F; 

class PFCandidateManagerAnalyzer: public BenchmarkAnalyzer, public PFCandidateManager {
 public:
  
  PFCandidateManagerAnalyzer(const edm::ParameterSet& parameterSet);
  //PFCandidateManagerAnalyzer(DQMStore::IBooker& b, const edm::ParameterSet& parameterSet);
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  //void beginJob() ;
  void beginJob(){} ;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  //void endJob();

 private:
  edm::EDGetTokenT< reco::PFCandidateCollection > myColl_;
  edm::EDGetTokenT< edm::View<reco::Candidate> > myMatchColl_;
  edm::InputTag matchLabel_;
};

#endif 
