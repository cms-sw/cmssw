#ifndef SiStripMonitorQuality_SiStripMonitorQuality_h
#define SiStripMonitorQuality_SiStripMonitorQuality_h
// -*- C++ -*-
//
// Package:     SiStripMonitorQuality
// Class  :     SiStripMonitorQuality
// 
/**\class SiStripMonitorQuality SiStripMonitorQuality.h 

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  dutta
//         Created:  Fri Dec  7 20:49:51 CET 2007
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

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "boost/cstdint.hpp"
#include <iostream>
#include <string>
#include <vector>

class MonitorElement;
class DQMStore;
class SiStripDetCabling;
class SiStripQuality;

class SiStripMonitorQuality : public DQMEDAnalyzer {
 public:
  explicit SiStripMonitorQuality(const edm::ParameterSet&);
  ~SiStripMonitorQuality();
  
  virtual void beginJob() ;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
  virtual void endJob() ;
  
  
 private:
  
  MonitorElement* getQualityME(uint32_t idet, const TrackerTopology* tTopo);
  
  
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  edm::ESHandle< SiStripDetCabling > detCabling_;
  edm::ESHandle< SiStripQuality > stripQuality_;

  std::map<uint32_t, MonitorElement*> QualityMEs;
  std::string dataLabel_;

  unsigned long long m_cacheID_;

};

#endif
