#ifndef DQM_SiPixelMonitorClient_SiPixelDaqInfo_H
#define DQM_SiPixelMonitorClient_SiPixelDaqInfo_H

// system include files
#include <fstream>
#include <iostream>
#include <memory>

// FWCore
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

class SiPixelDaqInfo : public DQMEDHarvester {
public:
  explicit SiPixelDaqInfo(const edm::ParameterSet &);
  ~SiPixelDaqInfo() override;

private:
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             const edm::LuminosityBlock &,
                             const edm::EventSetup &) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  MonitorElement *Fraction_;
  MonitorElement *FractionBarrel_;
  MonitorElement *FractionEndcap_;

  std::pair<int, int> FEDRange_;

  int NumberOfFeds_;

  int NEvents_;
  int nFEDsBarrel_;
  int nFEDsEndcap_;
  std::string daqSource_;
  int FEDs_[40];

  bool firstLumi;

  // define Token(-s)
  edm::EDGetTokenT<FEDRawDataCollection> daqSourceToken_;
};

#endif
