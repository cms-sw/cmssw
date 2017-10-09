#ifndef DQM_SiPixelMonitorClient_SiPixelCertification_H
#define DQM_SiPixelMonitorClient_SiPixelCertification_H

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class SiPixelCertification: public DQMEDHarvester{
public:
  explicit SiPixelCertification(const edm::ParameterSet&);
  ~SiPixelCertification();
  

private:
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, const edm::LuminosityBlock& , const  edm::EventSetup&) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  bool firstLumi;
  
  MonitorElement * CertificationPixel;
  MonitorElement * CertificationBarrel;
  MonitorElement * CertificationEndcap;

};

#endif
