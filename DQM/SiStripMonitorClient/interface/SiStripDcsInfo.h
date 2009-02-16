#ifndef SiStripMonitorClient_SiStripDcsInfo_h
#define SiStripMonitorClient_SiStripDcsInfo_h
// -*- C++ -*-
//
// Package:     SiStripMonitorClient
// Class  :     SiStripDcsInfo
// 
/**\class SiStripDcsInfo SiStripDcsInfo.h DQM/SiStripMonitorCluster/interface/SiStripDcsInfo.h

 Description: 
      Checks the # of SiStrip FEDs from DAQ
 Usage:
    <usage>

*/
//
//          Author:  Suchandra Dutta
//         Created:  Mon Feb 16 19:00:00 CET 2009
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

class SiStripDcsInfo: public edm::EDAnalyzer {

 public:

  /// Constructor
  SiStripDcsInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripDcsInfo();

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

  DQMStore* dqmStore_;
  MonitorElement * DcsFraction_;
  MonitorElement * DcsFractionTIB_;
  MonitorElement * DcsFractionTOB_;
  MonitorElement * DcsFractionTIDF_;
  MonitorElement * DcsFractionTIDB_;
  MonitorElement * DcsFractionTECF_;
  MonitorElement * DcsFractionTECB_;

  unsigned long long m_cacheID_;

};
#endif
