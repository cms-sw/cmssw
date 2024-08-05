// -*- C++ -*-
// Package:    EventFilter/Phase2PixelRawToDigi
// Class:      Phase2ITQCoreProducer
// Description: Make Phase2ITQCore objects for digis
// Maintainer: Si Hyun Jeon, shjeon@cern.ch
// Original Author:  Rohan Misra
// Created:  Thu, 30 Sep 2021 02:04:06 GMT
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITDigiHit.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITQCore.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITChip.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITChipBitStream.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

class Phase2ITQCoreProducer : public edm::stream::EDProducer<> {
public:
  Phase2ITQCoreProducer(const edm::ParameterSet&);
  ~Phase2ITQCoreProducer() override = default;

private:
  virtual void beginJob(const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  const edm::InputTag src_;
  const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> pixelDigi_token_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  typedef math::XYZPointD Point;
  typedef std::vector<Point> PointCollection;
};

Phase2ITQCoreProducer::Phase2ITQCoreProducer(const edm::ParameterSet& iConfig)
    : src_(iConfig.getParameter<edm::InputTag>("src")),
      pixelDigi_token_(consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("siPixelDigi"))),
      tTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()) {
  produces<edm::DetSetVector<Phase2ITQCore>>();
  produces<edm::DetSetVector<Phase2ITChipBitStream>>();
}

namespace {
  // Dimension for 2 chips module = 672 X 434 = 672 X (216 + 1 + 216 + 1)
  // Dimension for 4 chips module = 1354 X 434 = (672 + 5 + 672 + 5) X (216 + 1 + 216 + 1)
  // Spacing 1 in column and 5 in row is introduced for each chip in between
  // if neighboring chip exists
  const int kQCoresInChipRow = (672);
  const int kQCoresInChipColumn = (216);
  const int kQCoresInChipRowGap = (5);
  const int kQCoresInChipColumnGap = (10);
}  // namespace

Phase2ITDigiHit updateHitCoordinatesForLargePixels(Phase2ITDigiHit& hit) {
  /*
        In-place modification of Hit coordinates to take into account large pixels
        Hits corresponding to large pixels are remapped so they lie on the boundary of the chip
        Note that this operation can produce multiple hits with the same row/column coordinates
        Duplicates get removed later on
    */

  // Current values before remapping
  int row = hit.row();
  int col = hit.col();

  // Values after remapping
  int updated_row = 0;
  int updated_col = 0;

  // Remapping of the row coordinate
  if (row < kQCoresInChipRow) {
    updated_row = row;
  } else if (row < (kQCoresInChipRow + kQCoresInChipRowGap)) {
    updated_row = kQCoresInChipRow - 1;
  }  // This will be ignored for 2 chips module
  else if (row < (kQCoresInChipRow + 2 * kQCoresInChipRowGap)) {
    updated_row = kQCoresInChipRow;
  } else {
    updated_row = (hit.row() - 2 * kQCoresInChipRowGap);
  }

  // Remapping of the column coordinate
  if (col < kQCoresInChipColumn) {
    updated_col = col;
  } else if (col < kQCoresInChipColumn + kQCoresInChipColumnGap) {
    updated_col = kQCoresInChipColumn - kQCoresInChipColumnGap;
  } else if (col < (kQCoresInChipColumn + 2 * kQCoresInChipColumn)) {
    updated_col = kQCoresInChipColumn;
  } else {
    updated_col = (hit.col() - 2 * kQCoresInChipColumnGap);
  }

  hit.set_row(updated_row);
  hit.set_col(updated_col);

  return hit;
}

std::vector<Phase2ITDigiHit> adjustEdges(std::vector<Phase2ITDigiHit> hitList) {
  /*
        In-place modification of Hit coordinates to take into account large pixels
    */
  std::for_each(hitList.begin(), hitList.end(), &updateHitCoordinatesForLargePixels);
  return hitList;
}

std::vector<Phase2ITChip> splitByChip(std::vector<Phase2ITDigiHit> hitList) {
  // Split the hit list by read out chip
  std::vector<std::vector<Phase2ITDigiHit>> hits_per_chip(4);
  for (auto hit : hitList) {
    int chip_index = (hit.col() < kQCoresInChipColumn) ? 0 : 1;
    if (hit.row() >= kQCoresInChipRow) {
      chip_index += 2;
    }
    hits_per_chip[chip_index].push_back(hit);
  }

  // Generate Phase2ITChip objects from the hit lists
  std::vector<Phase2ITChip> chips;
  for (int chip_index = 0; chip_index < 4; chip_index++) {
    chips.push_back(Phase2ITChip(chip_index, hits_per_chip[chip_index]));
  }

  return chips;
}

std::vector<Phase2ITChip> processHits(std::vector<Phase2ITDigiHit> hitList) {
  std::vector<Phase2ITDigiHit> newHitList;
  newHitList = adjustEdges(hitList);

  std::vector<Phase2ITChip> chips = splitByChip(newHitList);
  std::vector<bool> code;

  for (size_t i = 0; i < chips.size(); i++) {
    Phase2ITChip chip = chips[i];
    code = chip.get_chip_code();
  }

  return chips;
}

// ------------ method called to produce the data  ------------
void Phase2ITQCoreProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  unique_ptr<edm::DetSetVector<Phase2ITQCore>> aQCoreVector = make_unique<edm::DetSetVector<Phase2ITQCore>>();
  unique_ptr<edm::DetSetVector<Phase2ITChipBitStream>> aBitStreamVector =
      make_unique<edm::DetSetVector<Phase2ITChipBitStream>>();

  auto const& tTopo = iSetup.getData(tTopoToken_);

  edm::Handle<edm::DetSetVector<PixelDigi>> pixelDigiHandle;
  iEvent.getByToken(pixelDigi_token_, pixelDigiHandle);

  edm::DetSetVector<PixelDigi>::const_iterator iterDet;
  for (iterDet = pixelDigiHandle->begin(); iterDet != pixelDigiHandle->end(); iterDet++) {
    DetId tkId = iterDet->id;
    edm::DetSet<PixelDigi> theDigis = (*pixelDigiHandle)[tkId];
    std::vector<Phase2ITDigiHit> hitlist;
    std::vector<int> id;

    if (tkId.subdetId() == PixelSubdetector::PixelBarrel) {
      int layer_num = tTopo.pxbLayer(tkId.rawId());
      int ladder_num = tTopo.pxbLadder(tkId.rawId());
      int module_num = tTopo.pxbModule(tkId.rawId());
      id = {tkId.subdetId(), layer_num, ladder_num, module_num};
    } else if (tkId.subdetId() == PixelSubdetector::PixelEndcap) {
      int module_num = tTopo.pxfModule(tkId());
      int disk_num = tTopo.pxfDisk(tkId());
      int blade_num = tTopo.pxfBlade(tkId());
      int panel_num = tTopo.pxfPanel(tkId());
      int side_num = tTopo.pxfSide(tkId());
      id = {tkId.subdetId(), module_num, disk_num, blade_num, panel_num, side_num};
    }

    for (auto iterDigi = theDigis.begin(); iterDigi != theDigis.end(); ++iterDigi) {
      hitlist.emplace_back(Phase2ITDigiHit(iterDigi->row(), iterDigi->column(), iterDigi->adc()));
    }

    std::vector<Phase2ITChip> chips = processHits(hitlist);

    DetSet<Phase2ITQCore> DetSetQCores(tkId);
    DetSet<Phase2ITChipBitStream> DetSetBitStream(tkId);

    for (size_t i = 0; i < chips.size(); i++) {
      Phase2ITChip chip = chips[i];
      std::vector<Phase2ITQCore> qcores = chip.get_organized_QCores();
      for (auto& qcore : qcores) {
        DetSetQCores.push_back(qcore);
      }

      Phase2ITChipBitStream aChipBitStream(i, chip.get_chip_code());
      DetSetBitStream.push_back(aChipBitStream);
    }

    aBitStreamVector->insert(DetSetBitStream);
    aQCoreVector->insert(DetSetQCores);
  }

  iEvent.put(std::move(aQCoreVector));
  iEvent.put(std::move(aBitStreamVector));
}

void Phase2ITQCoreProducer::beginJob(edm::EventSetup const&) {}

void Phase2ITQCoreProducer::endJob() {}

DEFINE_FWK_MODULE(Phase2ITQCoreProducer);
