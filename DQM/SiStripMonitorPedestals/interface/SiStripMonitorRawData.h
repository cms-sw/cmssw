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
//

// system include files
#include <memory>

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "boost/cstdint.hpp"
#include <iostream>
#include <string>
#include <vector>

class MonitorElement;
class DQMStore;
class SiStripDetCabling;

class SiStripMonitorRawData : public DQMEDAnalyzer {
 public:
  explicit SiStripMonitorRawData(const edm::ParameterSet&);
  ~SiStripMonitorRawData();
  
  virtual void beginJob() ;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  virtual void endJob() ;
  
  
 private:
  edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > digiToken_;

  MonitorElement* BadFedNumber;
  
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  edm::ESHandle< SiStripDetCabling > detcabling;
  std::vector<uint32_t> SelectedDetIds;

  unsigned long long m_cacheID_;

};

#endif
