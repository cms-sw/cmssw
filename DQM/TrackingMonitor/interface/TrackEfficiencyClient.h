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

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
//#include <map>

class DQMStore;

class TrackEfficiencyClient: public edm::EDAnalyzer {

 public:

  /// Constructor
  TrackEfficiencyClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~TrackEfficiencyClient();

 protected:

  /// BeginJob
  void beginJob(void);

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Analyze                                                                                                                                               
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);

  /// Endjob
  void endJob();
  
  /// EndRun
  void endRun();
 
  /// End Luminosity Block
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup);

 
 private:

  DQMStore* dqmStore_;
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
