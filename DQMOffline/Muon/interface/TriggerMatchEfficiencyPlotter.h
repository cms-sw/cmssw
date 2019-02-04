#ifndef TriggerMatchEfficiencyPlotter_H
#define TriggerMatchEfficiencyPlotter_H
/** \class TriggerMatch monitor
 *  *
 *   *  DQM monitoring source for Trigger matching efficiency plotter  feature added to miniAOD
 *    *
 *     *  \author Bibhuprasad Mahakud (Purdue University, West Lafayette, USA)
 *      */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "TH1F.h"

class TriggerMatchEfficiencyPlotter: public DQMEDHarvester {

public:

  /// Constructor
  TriggerMatchEfficiencyPlotter(const edm::ParameterSet& ps);
  
  /// Destructor
  ~TriggerMatchEfficiencyPlotter() override;

protected:

  /// DQM Client Diagnostic
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob


private:
  
  DQMStore* theDbe;
  edm::ParameterSet parameters;

  std::string triggerhistName1_;
  std::string triggerhistName2_;

  // efficiency histograms
  MonitorElement* h_eff_Path1_eta_tight;
  MonitorElement* h_eff_Path1_pt_tight;
  MonitorElement* h_eff_Path1_phi_tight;

  MonitorElement* h_eff_Path2_eta_tight;
  MonitorElement* h_eff_Path2_pt_tight;
  MonitorElement* h_eff_Path2_phi_tight;

  std::string theFolder;
};

#endif
