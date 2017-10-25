#ifndef DQM_SiPixelMonitorClient_SiPixelDcsInfo_H
#define DQM_SiPixelMonitorClient_SiPixelDcsInfo_H

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
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
#include "DQMServices/Core/interface/DQMEDHarvester.h"


class SiPixelDcsInfo : public DQMEDHarvester {
public:
  explicit SiPixelDcsInfo(const edm::ParameterSet&);
  ~SiPixelDcsInfo() override;
  

private:
  void dqmEndLuminosityBlock(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, const edm::LuminosityBlock& , const  edm::EventSetup&) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
  
  MonitorElement*  Fraction_;
  MonitorElement*  FractionBarrel_;
  MonitorElement*  FractionEndcap_;

  bool firstRun;
 
};

#endif
