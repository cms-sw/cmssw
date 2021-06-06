
#include "DQM/SiPixelMonitorClient/interface/SiPixelDaqInfo.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

using namespace std;
using namespace edm;
SiPixelDaqInfo::SiPixelDaqInfo(const edm::ParameterSet &ps) {
  FEDRange_.first = ps.getUntrackedParameter<unsigned int>("MinimumPixelFEDId", 0);
  FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumPixelFEDId", 39);
  daqSource_ = ps.getUntrackedParameter<string>("daqSource", "source");

  NumberOfFeds_ = FEDRange_.second - FEDRange_.first + 1;

  NEvents_ = 0;
  for (int i = 0; i != 40; i++)
    FEDs_[i] = 0;

  firstLumi = true;

  // set Token(-s)
  daqSourceToken_ = consumes<FEDRawDataCollection>(ps.getUntrackedParameter<string>("daqSource", "source"));
  runInfoToken_ = esConsumes<RunInfo, RunInfoRcd, edm::Transition::EndLuminosityBlock>();
}

SiPixelDaqInfo::~SiPixelDaqInfo() {}

void SiPixelDaqInfo::dqmEndLuminosityBlock(DQMStore::IBooker &iBooker,
                                           DQMStore::IGetter &iGetter,
                                           const edm::LuminosityBlock &lumiBlock,
                                           const edm::EventSetup &iSetup) {
  // Book somethings first time around
  if (firstLumi) {
    iBooker.setCurrentFolder("Pixel/EventInfo");
    Fraction_ = iBooker.bookFloat("DAQSummary");
    iBooker.setCurrentFolder("Pixel/EventInfo/DAQContents");
    FractionBarrel_ = iBooker.bookFloat("PixelBarrelFraction");
    FractionEndcap_ = iBooker.bookFloat("PixelEndcapFraction");

    firstLumi = false;
  }

  if (auto runInfoRec = iSetup.tryToGet<RunInfoRcd>()) {
    // get fed summary information
    const RunInfo &sumFED = runInfoRec->get(runInfoToken_);
    vector<int> FedsInIds = sumFED.m_fed_in;

    int FedCount = 0;
    int FedCountBarrel = 0;
    int FedCountEndcap = 0;

    // loop on all active feds
    for (unsigned int fedItr = 0; fedItr < FedsInIds.size(); ++fedItr) {
      int fedID = FedsInIds[fedItr];
      // make sure fed id is in allowed range
      if (fedID >= FEDRange_.first && fedID <= FEDRange_.second) {
        ++FedCount;
        if (fedID >= 0 && fedID <= 31)
          ++FedCountBarrel;
        else if (fedID >= 32 && fedID <= 39)
          ++FedCountEndcap;
      }
    }
    // Fill active fed fraction ME
    if (FedCountBarrel <= 32) {
      MonitorElement *mefed = iGetter.get("Pixel/EventInfo/DAQContents/fedcounter");
      FedCountBarrel = 0;
      FedCountEndcap = 0;
      FedCount = 0;
      NumberOfFeds_ = 40;
      if (mefed) {
        for (int i = 0; i != 40; i++) {
          if (i <= 31 && mefed->getBinContent(i + 1) > 0)
            FedCountBarrel++;
          if (i >= 32 && mefed->getBinContent(i + 1) > 0)
            FedCountEndcap++;
          if (mefed->getBinContent(i + 1) > 0)
            FedCount++;
        }
      }
    }
    if (NumberOfFeds_ > 0) {
      // all Pixel:
      Fraction_->Fill(FedCount / NumberOfFeds_);
      // Barrel:
      FractionBarrel_->Fill(FedCountBarrel / 32.);
      // Endcap:
      FractionEndcap_->Fill(FedCountEndcap / 8.);
    } else {
      Fraction_->Fill(-1);
      FractionBarrel_->Fill(-1);
      FractionEndcap_->Fill(-1);
    }
  } else {
    Fraction_->Fill(-1);
    FractionBarrel_->Fill(-1);
    FractionEndcap_->Fill(-1);
    return;
  }
}

void SiPixelDaqInfo::dqmEndJob(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter) {}
