#ifndef DQMOFFLINE_TRIGGER_TAU3MUMONITOR_H
#define DQMOFFLINE_TRIGGER_TAU3MUMONITOR_H

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMOffline/Trigger/interface/TriggerDQMBase.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

class Tau3MuMonitor : public DQMEDAnalyzer, public TriggerDQMBase {

 public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  Tau3MuMonitor(const edm::ParameterSet&);
  ~Tau3MuMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

 private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  // internally store a flag to remember whether the needed tau3mu collection is present and valid
  bool validProduct_ = true;

  edm::EDGetTokenT<reco::CompositeCandidateCollection> tauToken_;  // tau 3 mu collection

  MonitorElement* tau1DPt_;      // 1D tau pt histogram
  MonitorElement* tau1DEta_;     // 1D tau eta histogram
  MonitorElement* tau1DPhi_;     // 1D tau phi histogram
  MonitorElement* tau1DMass_;    // 1D tau mass histogram
  MonitorElement* tau2DEtaPhi_;  // 2D tau eta vs phi histogram

  MEbinning pt_binning_;    // for the 1D tau pt histogram
  MEbinning eta_binning_;   // for the 1D tau eta histogram and 2D tau eta vs phi histogram
  MEbinning phi_binning_;   // for the 1D tau phi histogram and 2D tau eta vs phi histogram
  MEbinning mass_binning_;  // for the 1D tau mass histogram

  std::unique_ptr<GenericTriggerEventFlag> genTriggerEventFlag_;
};

#endif
