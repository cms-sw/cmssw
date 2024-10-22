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
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
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

//DataFormats
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

class DQMDaqInfo : public edm::one::EDAnalyzer<> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;
  explicit DQMDaqInfo(const edm::ParameterSet&);
  ~DQMDaqInfo() override = default;

private:
  void beginJob() override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
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

DQMDaqInfo::DQMDaqInfo(const edm::ParameterSet& iConfig)
    : runInfoToken_{esConsumes<edm::Transition::BeginLuminosityBlock>()} {}

void DQMDaqInfo::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& iSetup) {
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));

  if (iSetup.tryToGet<RunInfoRcd>()) {
    if (auto sumFED = iSetup.getHandle(runInfoToken_)) {
      //const RunInfo* summaryFED=sumFED.product();

      std::vector<int> FedsInIds = sumFED->m_fed_in;

      float FedCount[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};

      for (int fedID : FedsInIds) {
        if (fedID >= PixelRange.first && fedID <= PixelRange.second)
          ++FedCount[Pixel];
        if (fedID >= TrackerRange.first && fedID <= TrackerRange.second)
          ++FedCount[SiStrip];
        if (fedID >= CSCRange.first && fedID <= CSCRange.second)
          ++FedCount[CSC];
        if (fedID >= RPCRange.first && fedID <= RPCRange.second)
          ++FedCount[RPC];
        if (fedID >= DTRange.first && fedID <= DTRange.second)
          ++FedCount[DT];
        if (fedID >= HcalRange.first && fedID <= HcalRange.second)
          ++FedCount[Hcal];
        if (fedID >= ECALBarrRange.first && fedID <= ECALBarrRange.second)
          ++FedCount[EcalBarrel];
        if ((fedID >= ECALEndcapRangeLow.first && fedID <= ECALEndcapRangeLow.second) ||
            (fedID >= ECALEndcapRangeHigh.first && fedID <= ECALEndcapRangeHigh.second))
          ++FedCount[EcalEndcap];
        if (fedID >= L1TRange.first && fedID <= L1TRange.second)
          ++FedCount[L1T];
      }

      for (int detIndex = 0; detIndex < 9; ++detIndex) {
        DaqFraction[detIndex]->Fill(FedCount[detIndex] / NumberOfFeds[detIndex]);
      }
    }
  } else {
    for (auto& detIndex : DaqFraction)
      detIndex->Fill(-1);
    return;
  }
}

void DQMDaqInfo::beginJob() {
  dbe_ = nullptr;
  dbe_ = edm::Service<DQMStore>().operator->();

  std::string commonFolder = "/EventInfo/DAQContents";
  std::string subsystFolder;
  std::string curentFolder;

  subsystFolder = "Pixel";
  curentFolder = subsystFolder + commonFolder;
  dbe_->setCurrentFolder(curentFolder);
  DaqFraction[Pixel] = dbe_->bookFloat("PixelDaqFraction");

  subsystFolder = "SiStrip";
  curentFolder = subsystFolder + commonFolder;
  dbe_->setCurrentFolder(curentFolder);
  DaqFraction[SiStrip] = dbe_->bookFloat("SiStripDaqFraction");

  subsystFolder = "RPC";
  curentFolder = subsystFolder + commonFolder;
  dbe_->setCurrentFolder(curentFolder);
  DaqFraction[RPC] = dbe_->bookFloat("RPCDaqFraction");

  subsystFolder = "CSC";
  curentFolder = subsystFolder + commonFolder;
  dbe_->setCurrentFolder(curentFolder);
  DaqFraction[CSC] = dbe_->bookFloat("CSCDaqFraction");

  subsystFolder = "DT";
  curentFolder = subsystFolder + commonFolder;
  dbe_->setCurrentFolder(curentFolder);
  DaqFraction[DT] = dbe_->bookFloat("DTDaqFraction");

  subsystFolder = "Hcal";
  curentFolder = subsystFolder + commonFolder;
  dbe_->setCurrentFolder(curentFolder);
  DaqFraction[Hcal] = dbe_->bookFloat("HcalDaqFraction");

  subsystFolder = "EcalBarrel";
  curentFolder = subsystFolder + commonFolder;
  dbe_->setCurrentFolder(curentFolder);
  DaqFraction[EcalBarrel] = dbe_->bookFloat("EcalBarrDaqFraction");

  subsystFolder = "EcalEndcap";
  curentFolder = subsystFolder + commonFolder;
  dbe_->setCurrentFolder(curentFolder);
  DaqFraction[EcalEndcap] = dbe_->bookFloat("EcalEndDaqFraction");

  subsystFolder = "L1T";
  curentFolder = subsystFolder + commonFolder;
  dbe_->setCurrentFolder(curentFolder);
  DaqFraction[L1T] = dbe_->bookFloat("L1TDaqFraction");

  PixelRange.first = FEDNumbering::MINSiPixelFEDID;
  PixelRange.second = FEDNumbering::MAXSiPixelFEDID;
  TrackerRange.first = FEDNumbering::MINSiStripFEDID;
  TrackerRange.second = FEDNumbering::MAXSiStripFEDID;
  CSCRange.first = FEDNumbering::MINCSCFEDID;
  CSCRange.second = FEDNumbering::MAXCSCFEDID;
  RPCRange.first = 790;
  RPCRange.second = 792;
  DTRange.first = 770;
  DTRange.second = 774;
  HcalRange.first = FEDNumbering::MINHCALFEDID;
  HcalRange.second = FEDNumbering::MAXHCALFEDID;
  L1TRange.first = FEDNumbering::MINTriggerGTPFEDID;
  L1TRange.second = FEDNumbering::MAXTriggerGTPFEDID;
  ECALBarrRange.first = 610;
  ECALBarrRange.second = 645;
  ECALEndcapRangeLow.first = 601;
  ECALEndcapRangeLow.second = 609;
  ECALEndcapRangeHigh.first = 646;
  ECALEndcapRangeHigh.second = 654;

  NumberOfFeds[Pixel] = PixelRange.second - PixelRange.first + 1;
  NumberOfFeds[SiStrip] = TrackerRange.second - TrackerRange.first + 1;
  NumberOfFeds[CSC] = CSCRange.second - CSCRange.first + 1;
  NumberOfFeds[RPC] = RPCRange.second - RPCRange.first + 1;
  NumberOfFeds[DT] = DTRange.second - DTRange.first + 1;
  NumberOfFeds[Hcal] = HcalRange.second - HcalRange.first + 1;
  NumberOfFeds[EcalBarrel] = ECALBarrRange.second - ECALBarrRange.first + 1;
  NumberOfFeds[EcalEndcap] = (ECALEndcapRangeLow.second - ECALEndcapRangeLow.first + 1) +
                             (ECALEndcapRangeHigh.second - ECALEndcapRangeHigh.first + 1);
  NumberOfFeds[L1T] = L1TRange.second - L1TRange.first + 1;
}

void DQMDaqInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMDaqInfo);
