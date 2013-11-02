#ifndef TrackingMonitorClient_TrackingCertificationInfo_h
#define TrackingMonitorClient_TrackingCertificationInfo_h
// -*- C++ -*-
//
// Package:     TrackingMonitorClient
// Class  :     TrackingCertificationInfo
// 
/**\class TrackingCertificationInfo TrackingCertificationInfo.h DQM/TrackingMonitorClient/interface/TrackingCertificationInfo.h

 Description: 

 Usage:
    <usage>

*/

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DQMStore;
class MonitorElement;
class SiStripDetCabling;

class TrackingCertificationInfo: public edm::EDAnalyzer {

 public:

  /// Constructor
  TrackingCertificationInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~TrackingCertificationInfo();

 private:

  /// BeginJob
  void beginJob();

  /// Begin Run
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// End Of Luminosity
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup);
  
  /// EndRun
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Analyze
  void analyze(edm::Event const&, edm::EventSetup const&);



private:

  void bookTrackingCertificationMEs();
  void bookTrackingCertificationMEsAtLumi();

  void resetTrackingCertificationMEs();
  void resetTrackingCertificationMEsAtLumi();

  void fillTrackingCertificationMEs(edm::EventSetup const& eSetup);
  void fillTrackingCertificationMEsAtLumi();

  void fillDummyTrackingCertification();
  void fillDummyTrackingCertificationAtLumi();


  DQMStore* dqmStore_;

  struct TrackingMEs{
    MonitorElement* TrackingFlag;
  };

  struct TrackingLSMEs{
    MonitorElement* TrackingFlag;
  };

  std::map<std::string, TrackingMEs>   TrackingMEsMap;
  std::map<std::string, TrackingLSMEs> TrackingLSMEsMap;

  MonitorElement * TrackingCertification;  
  MonitorElement * TrackingCertificationSummaryMap;  

  MonitorElement * TrackingLSCertification;  

  edm::ESHandle< SiStripDetCabling > detCabling_;
  edm::ParameterSet pSet_;

  bool trackingCertificationBooked_;
  bool trackingLSCertificationBooked_;
  int nFEDConnected_;
  bool allPixelFEDConnected_;
  std::string TopFolderName_;

  bool checkPixelFEDs_;

  unsigned long long m_cacheID_;

  std::vector<std::string> SubDetFolder;
};
#endif
