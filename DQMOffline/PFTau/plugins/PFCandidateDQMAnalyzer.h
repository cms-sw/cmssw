#ifndef __DQMOffline_PFTau_PFCandidateDQMAnalyzer__
#define __DQMOffline_PFTau_PFCandidateDQMAnalyzer__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMOffline/PFTau/interface/PFCandidateMonitor.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class PFCandidateDQMAnalyzer : public DQMEDAnalyzer {
public:
  PFCandidateDQMAnalyzer(const edm::ParameterSet &parameterSet);

private:
  void analyze(edm::Event const &, edm::EventSetup const &) override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  MonitorElement *eventId_;

  edm::EDGetTokenT<edm::View<reco::Candidate>> myCand_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> myMatchedCand_;
  edm::InputTag matchLabel_;
  edm::InputTag inputLabel_;
  std::string benchmarkLabel_;
  bool createEfficiencyHistos_;

  PFCandidateMonitor pfCandidateMonitor_;

  edm::ParameterSet pSet_;
  std::string eventInfoFolder_;
  std::string subsystemname_;

  int nBadEvents_;
};

#endif
