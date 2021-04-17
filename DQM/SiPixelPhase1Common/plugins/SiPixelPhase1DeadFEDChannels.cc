// -*- C++ -*-
//
// Package:    SiPixelPhase1DeadFEDChannels
// Class:      SiPixelPhase1DeadFEDChannels
//

// Original Author: F.Fiori

// C++ stuff
#include <iostream>

// CMSSW stuff
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"

// DQM Stuff
#include "DQMServices/Core/interface/DQMStore.h"

// Input data stuff
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"

// PixelDQM Framework
#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"

namespace {

  class SiPixelPhase1DeadFEDChannels final : public SiPixelPhase1Base {
    // List of quantities to be plotted.
    enum {
      DEADCHAN,
      DEADCHANROC

    };

  public:
    explicit SiPixelPhase1DeadFEDChannels(const edm::ParameterSet& conf);
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    edm::EDGetTokenT<PixelFEDChannelCollection> pixelFEDChannelCollectionToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
    edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> cablingMapToken_;

    bool firstEvent_;
    const TrackerGeometry* trackerGeometry_ = nullptr;
    const SiPixelFedCabling* cablingMap = nullptr;
  };

  SiPixelPhase1DeadFEDChannels::SiPixelPhase1DeadFEDChannels(const edm::ParameterSet& iConfig)
      : SiPixelPhase1Base(iConfig) {
    pixelFEDChannelCollectionToken_ = consumes<PixelFEDChannelCollection>(edm::InputTag("siPixelDigis"));
    trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
    cablingMapToken_ = esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd>();
    firstEvent_ = true;
  };

  void SiPixelPhase1DeadFEDChannels::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    if (!checktrigger(iEvent, iSetup, DCS))
      return;

    if (firstEvent_) {
      edm::ESHandle<TrackerGeometry> tmpTkGeometry = iSetup.getHandle(trackerGeomToken_);
      trackerGeometry_ = &(*tmpTkGeometry);

      edm::ESHandle<SiPixelFedCablingMap> pixelCabling = iSetup.getHandle(cablingMapToken_);
      cablingMap = pixelCabling.product();

      firstEvent_ = false;
    }

    edm::Handle<edmNew::DetSetVector<PixelFEDChannel> > input;

    iEvent.getByToken(pixelFEDChannelCollectionToken_, input);
    if (!input.isValid())
      return;

    for (const auto& disabledOnDetId : *input) {
      for (const auto& ch : disabledOnDetId) {
        sipixelobjects::CablingPathToDetUnit path = {ch.fed, ch.link, 0};

        for (path.roc = 1; path.roc <= (ch.roc_last - ch.roc_first) + 1; path.roc++) {
          const sipixelobjects::PixelROC* roc = cablingMap->findItem(path);
          assert(roc != nullptr);
          assert(roc->rawId() == disabledOnDetId.detId());

          const PixelGeomDetUnit* theGeomDet =
              dynamic_cast<const PixelGeomDetUnit*>(trackerGeometry_->idToDet(roc->rawId()));
          PixelTopology const* topology = &(theGeomDet->specificTopology());
          sipixelobjects::LocalPixel::RocRowCol local = {topology->rowsperroc() / 2,
                                                         topology->colsperroc() / 2};  //center of ROC
          sipixelobjects::GlobalPixel global = roc->toGlobal(sipixelobjects::LocalPixel(local));
          histo[DEADCHANROC].fill(disabledOnDetId.detId(), &iEvent, global.col, global.row);
        }

        histo[DEADCHAN].fill(disabledOnDetId.detId(), &iEvent);  // global count
      }
    }

    histo[DEADCHAN].executePerEventHarvesting(&iEvent);
  }
}  //namespace

DEFINE_FWK_MODULE(SiPixelPhase1DeadFEDChannels);
