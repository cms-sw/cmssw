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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DQMServices/Core/interface/DQMOneEDAnalyzer.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

class SiStripDetCabling;
class SiStripQuality;
class TrackerTopology;

class SiStripMonitorQuality : public DQMOneEDAnalyzer<> {
public:
  explicit SiStripMonitorQuality(const edm::ParameterSet &);
  ~SiStripMonitorQuality() override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void dqmEndRun(edm::Run const &run, edm::EventSetup const &eSetup) override;
  void endJob() override;

private:
  MonitorElement *getQualityME(uint32_t idet, const TrackerTopology *tTopo);

  DQMStore *dqmStore_;
  edm::ParameterSet conf_;
  edm::ESHandle<SiStripDetCabling> detCabling_;
  edm::ESHandle<SiStripQuality> stripQuality_;

  std::map<uint32_t, MonitorElement *> QualityMEs;
  std::string dataLabel_;

  unsigned long long m_cacheID_;
};

#endif
