#ifndef DQMSERVICES_COMPONENTS_DQMDaqInfo_H
#define DQMSERVICES_COMPONENTS_DQMDaqInfo_H
// -*- C++ -*-
//
// Package:    DQMDaqInfo
// Class:      DQMDaqInfo
//
/**\class DQMDaqInfo DQMDaqInfo.cc CondCore/DQMDaqInfo/src/DQMDaqInfo.cc
   
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ilaria SEGONI
//         Created:  Thu Sep 25 11:17:43 CEST 2008
//
//

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

//Run Info
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

//DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class DQMDaqInfo : public edm::EDAnalyzer {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;
  explicit DQMDaqInfo(const edm::ParameterSet&);
  ~DQMDaqInfo() override;

private:
  void beginJob() override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  DQMStore* dbe_;

  enum subDetList { Pixel, SiStrip, EcalBarrel, EcalEndcap, Hcal, DT, CSC, RPC, L1T };

  MonitorElement* DaqFraction[9];

  std::pair<int, int> PixelRange;
  std::pair<int, int> TrackerRange;
  std::pair<int, int> CSCRange;
  std::pair<int, int> RPCRange;
  std::pair<int, int> DTRange;
  std::pair<int, int> HcalRange;
  std::pair<int, int> ECALBarrRange;
  std::pair<int, int> ECALEndcapRangeLow;
  std::pair<int, int> ECALEndcapRangeHigh;
  std::pair<int, int> L1TRange;

  float NumberOfFeds[9];
};

#endif
