#ifndef DQMOffline_Trigger_TagAndProbeBtagTriggerMonitor_H
#define DQMOffline_Trigger_TagAndProbeBtagTriggerMonitor_H

/*
  TagAndProbeBtagTriggerMonitor DQM code
*/
//
// Originally created by:  Roberval Walsh
//                         June 2017

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

class TagAndProbeBtagTriggerMonitor : public DQMEDAnalyzer {

 public:
  TagAndProbeBtagTriggerMonitor(const edm::ParameterSet&);
  ~TagAndProbeBtagTriggerMonitor() throw() override;

 protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

 private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  std::string processname_;
  std::string triggerobjbtag_;

  double jetPtmin_;
  double jetEtamax_;
  double tagBtagmin_;
  double probeBtagmin_;

  std::vector<double> jetPtbins_;
  std::vector<double> jetEtabins_;
  std::vector<double> jetPhibins_;
  std::vector<double> jetBtagbins_;

  edm::InputTag triggerSummaryLabel_;

  edm::EDGetTokenT<reco::JetTagCollection> offlineBtagToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerSummaryToken_;

  MonitorElement* pt_jet1_;
  MonitorElement* pt_jet2_;
  MonitorElement* eta_jet1_;
  MonitorElement* eta_jet2_;
  MonitorElement* phi_jet1_;
  MonitorElement* phi_jet2_;
  MonitorElement* eta_phi_jet1_;
  MonitorElement* eta_phi_jet2_;

  MonitorElement* pt_probe_;
  MonitorElement* pt_probe_match_;
  MonitorElement* eta_probe_;
  MonitorElement* eta_probe_match_;
  MonitorElement* phi_probe_;
  MonitorElement* phi_probe_match_;
  MonitorElement* eta_phi_probe_;
  MonitorElement* eta_phi_probe_match_;

  MonitorElement* discr_offline_btag_jet1_;
  MonitorElement* discr_offline_btag_jet2_;

  std::unique_ptr<GenericTriggerEventFlag> genTriggerEventFlag_;  // tag & probe: trigger flag for num and den
};

#endif
