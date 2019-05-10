#ifndef DQM_SiPixelMonitorClient_SiPixelCertification_H
#define DQM_SiPixelMonitorClient_SiPixelCertification_H

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
#include "DQMServices/Core/interface/MonitorElement.h"

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
