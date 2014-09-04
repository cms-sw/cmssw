#ifndef HcalMonitorClient_HcalDcsInfo_h
#define HcalMonitorClient_HcalDcsInfo_h
// -*- C++ -*-
//
// Package:     HcalMonitorClient
// Class  :     HcalDCSInfo
// 
/**\class HcalDCSInfo HcalDCSInfo.h DQM/HcalMonitorCluster/interface/HcalDCSInfo.h

 Description: 
      Checks the # of Hcal FEDs from DAQ
 Usage:
    <usage>

*/
//         Author:  Jeff Temple
//         Created:  Fri Mar 6 00:15:00 CET 2009
//
//          based on v1.1 of DQM/SiStripMonitorClient/src/SiStripDCsInfo.cc
//          by:  Suchandra Dutta
//         Created:  Mon Feb 16 19:00:00 CET 2009
//

#include <string>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
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

class MonitorElement;

class HcalDCSInfo: public DQMEDAnalyzer {

 public:

  /// Constructor
  HcalDCSInfo(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~HcalDCSInfo();
  
  /// BeginJob
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);


 private:

  /// End Luminosity Block
  //virtual void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) ;

  /// Analyze
  void analyze(edm::Event const&, edm::EventSetup const&);

  int debug_;

private:

  MonitorElement * DCSSummaryMap_;
  MonitorElement * DCSFraction_;
  MonitorElement * DCSFractionHB_;
  MonitorElement * DCSFractionHE_;
  MonitorElement * DCSFractionHO_;
  MonitorElement * DCSFractionHF_;
  MonitorElement * DCSFractionHO0_;
  MonitorElement * DCSFractionHO12_;
  MonitorElement * DCSFractionHFlumi_;

  unsigned long long m_cacheID_;
  std::string rootFolder_;

};
#endif


