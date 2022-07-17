#ifndef __DQMOffline_PFTau_PFMETDQMAnalyzer__
#define __DQMOffline_PFTau_PFMETDQMAnalyzer__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMOffline/PFTau/interface/PFMETMonitor.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class PFMETDQMAnalyzer : public DQMEDAnalyzer {
public:
  PFMETDQMAnalyzer(const edm::ParameterSet &parameterSet);

private:
  void analyze(edm::Event const &, edm::EventSetup const &) override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  edm::EDGetTokenT<edm::View<reco::MET>> myMET_;
  edm::EDGetTokenT<edm::View<reco::MET>> myMatchedMET_;
  edm::InputTag matchLabel_;
  edm::InputTag inputLabel_;
  std::string benchmarkLabel_;

  PFMETMonitor pfMETMonitor_;

  edm::ParameterSet pSet_;
  std::string eventInfoFolder_;
  std::string subsystemname_;

  int nBadEvents_;
};

#endif
