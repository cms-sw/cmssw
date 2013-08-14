#ifndef SiStripMonitorRawData_SiStripMonitorRawData_h
#define SiStripMonitorRawData_SiStripMonitorRawData_h
// -*- C++ -*-
//
// Package:     SiStripMonitorRawData
// Class  :     SiStripMonitorRawData
// 
/**\class SiStripMonitorRawData SiStripMonitorRawData.h 

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  dutta
//         Created:  Sat Feb  4 20:49:51 CET 2006
// $Id: SiStripMonitorRawData.h,v 1.5 2009/11/05 21:08:29 dutta Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "boost/cstdint.hpp"
#include <iostream>
#include <string>
#include <vector>

class MonitorElement;
class DQMStore;
class SiStripDetCabling;

class SiStripMonitorRawData : public edm::EDAnalyzer {
 public:
  explicit SiStripMonitorRawData(const edm::ParameterSet&);
  ~SiStripMonitorRawData();
  
  virtual void beginJob() ;
  virtual void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  virtual void endJob() ;
  
  
 private:
  MonitorElement* BadFedNumber;
  
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  edm::ESHandle< SiStripDetCabling > detcabling;
  std::vector<uint32_t> SelectedDetIds;

  unsigned long long m_cacheID_;

};

#endif
