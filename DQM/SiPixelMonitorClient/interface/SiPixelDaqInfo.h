#ifndef DQM_SiPixelMonitorClient_SiPixelDaqInfo_H
#define DQM_SiPixelMonitorClient_SiPixelDaqInfo_H

// system include files
#include <fstream>
#include <iostream>
#include <memory>

// DQM
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
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
  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
};

#endif
