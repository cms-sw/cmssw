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

class SiStripCertificationInfo: public edm::EDAnalyzer {

 public:

  /// Constructor
  SiStripCertificationInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripCertificationInfo();

 private:

  /// BeginJob
  void beginJob(edm::EventSetup const& eSetup);

  /// Begin Run
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Begin Of Luminosity
                                                                               
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup);

  /// Analyze

  void analyze(edm::Event const&, edm::EventSetup const&);



private:

  void bookSiStripCertificationMEs();
  void resetSiStripCertificationMEs();
  void fillSiStripCertificationMEs();

  void bookTrackingCertificationMEs();
  void resetTrackingCertificationMEs();
  void fillTrackingCertificationMEs();

  DQMStore* dqmStore_;
  MonitorElement * SiStripCertification;
  MonitorElement * SiStripCertificationTIB;
  MonitorElement * SiStripCertificationTOB;
  MonitorElement * SiStripCertificationTIDF;
  MonitorElement * SiStripCertificationTIDB;
  MonitorElement * SiStripCertificationTECF;
  MonitorElement * SiStripCertificationTECB;

  
  MonitorElement * TrackingCertification;  
  MonitorElement * TrackingCertificationRate;
  MonitorElement * TrackingCertificationChi2overDoF;
  MonitorElement * TrackingCertificationRecHits;

  bool trackingCertificationBooked_;
  bool sistripCertificationBooked_;
  unsigned long long m_cacheID_;

};
#endif
