#ifndef ElectronEfficiencyPlotter_H
#define ElectronEfficiencyPlotter_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "TH1F.h"

class ElectronEfficiencyPlotter : public DQMEDHarvester{
public:
  // Constructor
  ElectronEfficiencyPlotter(const edm::ParameterSet& ps);
  // Destructor
  ~ElectronEfficiencyPlotter() override;

protected:
  // DQM Client Diagnostic
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  // counters
  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;

  edm::ParameterSet parameters;

  int ptBin;
  double ptMin;
  double ptMax;

  std::string ID_;

  MonitorElement* h_eff_pt_EB_ID;
  MonitorElement* h_eff_pt_EE_ID;
  MonitorElement* h_eff_pt_EB_HLT;
  MonitorElement* h_eff_pt_EE_HLT;

  std::string theFolder_;
  std::string sourceFolder_;
  void GetEfficiency(MonitorElement* Numerator, MonitorElement* Denominator, MonitorElement* Efficiency);

};

#endif
