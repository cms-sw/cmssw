#ifndef SiStripMonitorClient_SiStripCertificationInfo_h
#define SiStripMonitorClient_SiStripCertificationInfo_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripCertificationInfo
// 
/**\class SiStripCertificationInfo SiStripCertificationInfo.h DQM/SiStripMonitorCluster/interface/SiStripCertificationInfo.h

 Description: 
      Checks the # of SiStrip FEDs from DAQ
 Usage:
    <usage>

*/
//
//          Author:  Suchandra Dutta
//         reated:  Mon Feb 16 19200:00 CET 2009
//

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

class SiStripCertificationInfo: public edm::EDAnalyzer {

 public:

  /// Constructor
  SiStripCertificationInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripCertificationInfo();

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

  void bookSiStripCertificationMEs();
  void resetSiStripCertificationMEs();
  void fillSiStripCertificationMEs(edm::EventSetup const& eSetup);

  void bookTrackingCertificationMEs();
  void resetTrackingCertificationMEs();
  void fillTrackingCertificationMEs(edm::EventSetup const& eSetup);

  void fillDummySiStripCertification();
  void fillDummyTrackingCertification();

  void fillSiStripCertificationMEsAtLumi();

  DQMStore* dqmStore_;



  struct SubDetMEs{
    MonitorElement* det_fractionME;
    std::string folder_name;
    std::string subdet_tag;
    int n_layer;
  };

  MonitorElement * SiStripCertification;
  MonitorElement * SiStripCertificationMap; 
  std::map<std::string, SubDetMEs> SubDetMEsMap;
  std::map<std::string, MonitorElement*> TrackingMEsMap;
  MonitorElement * SiStripCertificationSummaryMap;

  MonitorElement * TrackingCertification;  

  bool trackingCertificationBooked_;
  bool sistripCertificationBooked_;
  unsigned long long m_cacheID_;

  edm::ESHandle< SiStripDetCabling > detCabling_;

  int nFEDConnected_;
};
#endif
