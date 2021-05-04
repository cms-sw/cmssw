#ifndef DQM_SiPixelMonitorClient_SiPixelDcsInfo_H
#define DQM_SiPixelMonitorClient_SiPixelDcsInfo_H

// system include files
#include <fstream>
#include <iostream>
#include <memory>

// DQM
#include "DQMServices/Core/interface/DQMEDHarvester.h"

class SiPixelDcsInfo : public DQMEDHarvester {
public:
  explicit SiPixelDcsInfo(const edm::ParameterSet &);
  ~SiPixelDcsInfo() override;

private:
  void dqmEndLuminosityBlock(DQMStore::IBooker &iBooker,
                             DQMStore::IGetter &iGetter,
                             const edm::LuminosityBlock &,
                             const edm::EventSetup &) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  MonitorElement *Fraction_;
  MonitorElement *FractionBarrel_;
  MonitorElement *FractionEndcap_;

  bool firstRun;
};

#endif
