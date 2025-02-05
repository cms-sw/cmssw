#ifndef HLTriggerOffline_Scouting_ElectronEfficiencyPlotter_H
#define HLTriggerOffline_Scouting_ElectronEfficiencyPlotter_H

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class ElectronEfficiencyPlotter : public DQMEDHarvester {
public:
  // Constructor
  ElectronEfficiencyPlotter(const edm::ParameterSet& ps);
  // Destructor
  ~ElectronEfficiencyPlotter() override = default;

protected:
  // DQM Client Diagnostic
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  // counters

  int ptBin_;
  double ptMin_;
  double ptMax_;

  std::string ID_;

  MonitorElement* h_eff_pt_EB_doubleEG_HLT;
  MonitorElement* h_eff_pt_EE_doubleEG_HLT;
  MonitorElement* h_eff_pt_EB_singlePhoton_HLT;
  MonitorElement* h_eff_pt_EE_singlePhoton_HLT;

  std::string theFolder_;
  std::string sourceFolder_;
  void calculateEfficiency(MonitorElement* Numerator, MonitorElement* Denominator, MonitorElement* Efficiency);
};

#endif
