#ifndef __DQMOffline_PFTau_PFMuonDQMAnalyzer__
#define __DQMOffline_PFTau_PFMuonDQMAnalyzer__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMOffline/PFTau/interface/PFCandidateMonitor.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class PFMuonDQMAnalyzer : public DQMEDAnalyzer {
public:
  PFMuonDQMAnalyzer(const edm::ParameterSet &parameterSet);

private:
  void analyze(edm::Event const &, edm::EventSetup const &) override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  edm::EDGetTokenT<edm::View<reco::Muon>> myCand_;
  edm::EDGetTokenT<edm::View<reco::Muon>> myMatchedCand_;
  edm::InputTag matchLabel_;
  edm::InputTag inputLabel_;
  std::string benchmarkLabel_;
  bool createEfficiencyHistos_;

  double ptBase_;
  double ptNotPF_;

  PFCandidateMonitor pfCandidateMonitor_;

  edm::ParameterSet pSet_;
  std::string eventInfoFolder_;
  std::string subsystemname_;

  int nBadEvents_;
};

#endif
