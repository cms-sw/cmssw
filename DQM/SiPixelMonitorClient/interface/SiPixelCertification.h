#ifndef DQM_SiPixelMonitorClient_SiPixelCertification_H
#define DQM_SiPixelMonitorClient_SiPixelCertification_H

// system include files
#include <fstream>
#include <iostream>
#include <memory>

// DQM
#include "DQMServices/Core/interface/DQMEDHarvester.h"

class SiPixelCertification : public DQMEDHarvester {
public:
  explicit SiPixelCertification(const edm::ParameterSet &);
  ~SiPixelCertification() override;

private:
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             const edm::LuminosityBlock &,
                             const edm::EventSetup &) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  bool firstLumi;

  MonitorElement *CertificationPixel;
  MonitorElement *CertificationBarrel;
  MonitorElement *CertificationEndcap;
};

#endif
