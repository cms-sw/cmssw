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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include <DQMServices/Core/interface/DQMOneEDAnalyzer.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

class SiStripDetCabling;

class SiStripMonitorRawData : public DQMOneEDAnalyzer<> {
public:
  explicit SiStripMonitorRawData(const edm::ParameterSet &);
  ~SiStripMonitorRawData() override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void dqmEndRun(edm::Run const &run, edm::EventSetup const &eSetup) override;
  void endJob() override;

private:
  edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi>> digiToken_;

  MonitorElement *BadFedNumber;

  DQMStore *dqmStore_;
  edm::ParameterSet conf_;
  edm::ESHandle<SiStripDetCabling> detcabling;
  std::vector<uint32_t> SelectedDetIds;

  unsigned long long m_cacheID_;
};

#endif
