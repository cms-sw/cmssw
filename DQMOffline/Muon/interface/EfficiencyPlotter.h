#ifndef EfficiencyPlotter_H
#define EfficiencyPlotter_H


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

class EfficiencyPlotter: public DQMEDHarvester {

public:

  /// Constructor
  EfficiencyPlotter(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~EfficiencyPlotter();

protected:

  /// DQM Client Diagnostic
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob


private:

  // counters
  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;
  int run;
  // Switch for verbosity
  std::string metname;

  DQMStore* theDbe;
  edm::ParameterSet parameters;

   //histo binning parameters
  int etaBin;
  double etaMin;
  double etaMax;

  int ptBin;
  double ptMin;
  double ptMax;

  int phiBin;
  double phiMin;
  double phiMax;
  
  int vtxBin;
  double vtxMin;
  double vtxMax;
  
  std::string ID_;

  // efficiency histograms
  MonitorElement* h_eff_pt_ID;
  MonitorElement* h_eff_pt_EB_ID;
  MonitorElement* h_eff_pt_EE_ID;
  MonitorElement* h_eff_eta_ID;
  MonitorElement* h_eff_hp_eta_ID;
  MonitorElement* h_eff_phi_ID;
  MonitorElement* h_eff_pt_detIsoID;
  MonitorElement* h_eff_pt_EB_detIsoID;
  MonitorElement* h_eff_pt_EE_detIsoID;
  MonitorElement* h_eff_pt_pfIsoID;
  MonitorElement* h_eff_pt_EB_pfIsoID;
  MonitorElement* h_eff_pt_EE_pfIsoID;

  MonitorElement* h_eff_vtx_detIsoID;
  MonitorElement* h_eff_vtx_pfIsoID;
  MonitorElement* h_eff_vtx_EB_detIsoID;
  MonitorElement* h_eff_vtx_EE_detIsoID;
  MonitorElement* h_eff_vtx_EB_pfIsoID;
  MonitorElement* h_eff_vtx_EE_pfIsoID;

  MonitorElement* h_eff_pt_pfIsodBID;
  MonitorElement* h_eff_pt_EB_pfIsodBID;
  MonitorElement* h_eff_pt_EE_pfIsodBID;

  MonitorElement* h_eff_vtx_pfIsodBID;
  MonitorElement* h_eff_vtx_EB_pfIsodBID;
  MonitorElement* h_eff_vtx_EE_pfIsodBID;
};

#endif
