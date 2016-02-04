#ifndef __DQMOffline_PFTau_PFCandidateDQMAnalyzer__
#define __DQMOffline_PFTau_PFCandidateDQMAnalyzer__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMOffline/PFTau/interface/PFCandidateMonitor.h"


class PFCandidateDQMAnalyzer: public edm::EDAnalyzer {
 public:
  
  PFCandidateDQMAnalyzer(const edm::ParameterSet& parameterSet);
  
 private:
  void analyze(edm::Event const&, edm::EventSetup const&);
  void beginJob() ;
  void endJob();

  void storeBadEvents(edm::Event const&, float& val);

  edm::InputTag matchLabel_;
  edm::InputTag inputLabel_;
  std::string benchmarkLabel_;
  
  PFCandidateMonitor pfCandidateMonitor_;

  edm::ParameterSet pSet_;

  int nBadEvents_;
};

#endif 
