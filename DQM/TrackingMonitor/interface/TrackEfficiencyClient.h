#ifndef TrackingMonitor_TrackEfficiencyClient_h
#define TrackingMonitor_TrackEfficiencyClient_h
// -*- C++ -*-
//
// Package:    TrackingMonitor
// Class  :    TrackEfficiencyClient
// 
/**\class TrackEfficiencyClient TrackEfficiencyClient.h DQM/TrackingMonitor/interface/TrackEfficiencyClient.h
DQM class to compute the tracking efficiency 
*/
// Original Author:  A.-C. Le Bihan
//         Created:  Fri Dec  5 12:14:22 CET 2008


#include <string>

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
//#include <map>

class TrackEfficiencyClient: public DQMEDHarvester
{

 public:

  /// Constructor
  TrackEfficiencyClient(const edm::ParameterSet& ps);
  
  /// Destructor
  ~TrackEfficiencyClient() override;

 protected:

  /// BeginJob
  void beginJob(void) override;

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  /// EndJob
  void dqmEndJob(DQMStore::IBooker & ibooker_, DQMStore::IGetter & igetter_) override;

 private:

  /// book MEs
  void bookMEs(DQMStore::IBooker & ibooker_);

  edm::ParameterSet conf_;
  
  bool trackEfficiency_; //1 if one wants to measure the tracking efficiency
                         //0 if one wants to measure the muon reco efficiency
  
  std::string histName;  
  std::string algoName_;
  std::string FolderName_;
  
  MonitorElement * effX;
  MonitorElement * effY;  
  MonitorElement * effZ; 
  MonitorElement * effEta;
  MonitorElement * effPhi;
  MonitorElement * effD0; 
  MonitorElement * effCompatibleLayers; 
 
};
#endif
