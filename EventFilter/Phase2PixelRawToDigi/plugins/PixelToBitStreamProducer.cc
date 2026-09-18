// -*- C++ -*-
// Package:    EventFilter/Phase2PixelRawToDigi
// Class:      PixelToBitStreamProducer
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
#include "DataFormats/Phase2TrackerDigi/interface/ChipModuleMap.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

using namespace Phase2DAQFormatSpecification;

// Inter-chip gap-pixel handling. The chip cannot physically read out pixels in the column gap (216..217) or row gap (672..681);
// Each mode below picks how the encoder treats them:
//   DROP      : discard them (matches real chip output) so that input PixelDigi can be fully restored
//   KEEP      : route to a chip whose offsets give them addressable qrow/ccol values, so the decoder reconstructs the exact (row, col, adc). Lossless but non-physical.
//   AGGREGATE : merge the gap pixel's ADC into the nearest chip-edge pixel (saturated at 15), then drop it. Models the real chip's "large pixel" charge sharing.
//   FIXME later with proper simulation of large pixels, this needs to be rewritten
enum class GapMode { Drop, Keep, Aggregate };

namespace {
  GapMode parseGapMode(const std::string& s) {
    if (s == "DROP")
      return GapMode::Drop;
    if (s == "KEEP")
      return GapMode::Keep;
    if (s == "AGGREGATE")
      return GapMode::Aggregate;
    throw cms::Exception("PixelToBitStreamProducer")
        << "handleGapPixels must be one of DROP/KEEP/AGGREGATE, got '" << s << "'";
  }
}  // namespace

class PixelToBitStreamProducer : public edm::stream::EDProducer<> {
public:
  PixelToBitStreamProducer(const edm::ParameterSet&);
  ~PixelToBitStreamProducer() override = default;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::InputTag src_;
  const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> pixelDigiToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  // Cabling map supplies the per-module Module_SubType that keys the ChipMap
  // chip-index convention.
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
  const GapMode gapMode_;
  // Drop the per-hit 4-bit ToT field (RD53B "Drop ToT" / binary readout mode).
  // Default false = ToTs included.
  const bool dropTot_;
  typedef math::XYZPointD Point;
  typedef std::vector<Point> PointCollection;
};

PixelToBitStreamProducer::PixelToBitStreamProducer(const edm::ParameterSet& iConfig)
    : src_(iConfig.getParameter<edm::InputTag>("src")),
      pixelDigiToken_(consumes(iConfig.getParameter<edm::InputTag>("siPixelDigi"))),
      tTopoToken_(esConsumes()),
      cablingMapToken_(esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd>()),
      gapMode_(parseGapMode(iConfig.getUntrackedParameter<std::string>("handleGapPixels", "AGGREGATE"))),
      dropTot_(iConfig.getUntrackedParameter<bool>("dropTot", false)) {
  produces<edm::DetSetVector<Phase2ITQCore>>();
  produces<edm::DetSetVector<Phase2ITChipBitStream>>();
}

namespace {
  // Module-coordinate geometry (single source of truth lives in Phase2ITChip.h).
  // Dimension for 1 chip module (PXB Layer1, 3dim) = 672 X 216
  // Dimension for 2 chips module = 672 X 434 = 672 X (216 + 2 + 216)
  // Dimension for 4 chips module = 1354 X 434 = (672 + 10 + 672) X (216 + 2 + 216)
  constexpr int kRowsPerChip = Phase2ITChip::kRowsPerChip;          // 672
  constexpr int kColsPerChip = Phase2ITChip::kColsPerChip;          // 216
  constexpr int kLargePixelNRows = Phase2ITChip::kLargePixelNRows;  // 10
  constexpr int kLargePixelNCols = Phase2ITChip::kLargePixelNCols;  // 2
}  // namespace

bool isInGapRegion(const Phase2ITDigiHit& hit) {
  int r = hit.row();
  int c = hit.col();
  bool rowInGap = (r >= kRowsPerChip) && (r < kRowsPerChip + kLargePixelNRows);
  bool colInGap = (c >= kColsPerChip) && (c < kColsPerChip + kLargePixelNCols);
  return rowInGap || colInGap;
}

// Merge gap-pixel ADCs into the nearest chip-edge pixel, then drop them.
// Models the real chip's "large pixel" charge sharing: a hit in the gap shares charge with the chip-edge pixel it's bonded to.
std::vector<Phase2ITDigiHit> aggregateGap(const std::vector<Phase2ITDigiHit>& hitList) {
  // Routing rule: gap col 216 -> col 215, col 217 -> col 218; row 672..676
  // -> row 671, row 677..681 -> row 682. Corner gap uses both.
  auto edgeCol = [](int c) {
    if (c == kColsPerChip)
      return kColsPerChip - 1;
    if (c == kColsPerChip + 1)
      return kColsPerChip + kLargePixelNCols;
    return c;
  };
  auto edgeRow = [](int r) {
    if (r >= kRowsPerChip && r < kRowsPerChip + kLargePixelNRows / 2)
      return kRowsPerChip - 1;
    if (r >= kRowsPerChip + kLargePixelNRows / 2 && r < kRowsPerChip + kLargePixelNRows)
      return kRowsPerChip + kLargePixelNRows;
    return r;
  };
  std::map<std::pair<int, int>, int> adcByPos;  // position -> accumulated ADC
  for (const auto& h : hitList) {
    int r = edgeRow(h.row());
    int c = edgeCol(h.col());
    auto& a = adcByPos[{r, c}];
    a = std::min(15, a + h.adc());  // saturate at 4-bit max
  }
  std::vector<Phase2ITDigiHit> out;
  out.reserve(adcByPos.size());
  for (auto& kv : adcByPos)
    out.emplace_back(kv.first.first, kv.first.second, kv.second);
  return out;
}

// Split a module-coordinate hit list into per-chip lists in chip-relative coordinates.
// The chip boundary depends on the gap mode: standard geometry uses physical chip extents,
// KEEP uses the gap-midline geometry so every pixel (including pixels in gaps) belongs to one chip in order to not drop those .
std::vector<Phase2ITChip> splitByChip(const std::vector<Phase2ITDigiHit>& hitList,
                                      GapMode mode,
                                      int subtype,
                                      uint32_t detId = 0) {
  const bool keep = (mode == GapMode::Keep);
  const int rowsPerChip = keep ? Phase2ITChip::kRowsPerChipKeep : kRowsPerChip;
  const int largePixelNRows = keep ? Phase2ITChip::kLargePixelNRowsKeep : kLargePixelNRows;
  const int colsPerChip = keep ? Phase2ITChip::kColsPerChipKeep : kColsPerChip;
  const int largePixelNCols = keep ? Phase2ITChip::kLargePixelNColsKeep : kLargePixelNCols;

  // Chip-index convention: ChipModuleMap quadrant table keyed by Module_SubType.
  const int nChips = ChipModuleMap::nChips(subtype);
  const int nChipRows = (nChips == 4) ? 2 : 1;
  const int nChipCols = (nChips == 1) ? 1 : 2;

  std::array<std::vector<Phase2ITDigiHit>, CHIPS_PER_MODULE> hitsPerChip;
  for (auto hit : hitList) {
    // Physical half along each axis (0 = low, 1 = high). The coordinate offset
    // follows the physical half, NOT the chip index, so it stays correct under
    // any chip-numbering convention.
    const int rowHalf = (hit.row() >= rowsPerChip) ? 1 : 0;
    const int colHalf = (hit.col() >= colsPerChip) ? 1 : 0;
    auto q = ChipModuleMap::gridToQuadrant(rowHalf, colHalf, nChipRows, nChipCols);
    const int chipIndex = ChipModuleMap::chipIndex(subtype, q.first, q.second);
    const int rowOffset = rowHalf ? (rowsPerChip + largePixelNRows) : 0;
    const int colOffset = colHalf ? (colsPerChip + largePixelNCols) : 0;
    hit.set_row(hit.row() - rowOffset);
    hit.set_col(hit.col() - colOffset);
    hitsPerChip[chipIndex].push_back(hit);
  }

  // Emit exactly nChips streams (cabling-map count via Module_SubType).
  std::vector<Phase2ITChip> chips;
  chips.reserve(nChips);
  for (int chipIndex = 0; chipIndex < nChips; chipIndex++)
    chips.emplace_back(chipIndex, hitsPerChip[chipIndex], detId);
  return chips;
}

std::vector<Phase2ITChip> processHits(std::vector<Phase2ITDigiHit> hitList,
                                      GapMode mode,
                                      int subtype,
                                      uint32_t detId = 0) {
  switch (mode) {
    case GapMode::Drop:
      // Discard gap-region hits entirely.
      hitList.erase(std::remove_if(hitList.begin(), hitList.end(), &isInGapRegion), hitList.end());
      break;
    case GapMode::Keep:
      // Leave gap pixels in the list.
      break;
    case GapMode::Aggregate:
      // Merge gap-pixel ADCs into the nearest chip-edge pixel.
      hitList = aggregateGap(hitList);
      break;
  }
  return splitByChip(hitList, mode, subtype, detId);
}

void PixelToBitStreamProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  unique_ptr<edm::DetSetVector<Phase2ITQCore>> aQCoreVector = make_unique<edm::DetSetVector<Phase2ITQCore>>();
  unique_ptr<edm::DetSetVector<Phase2ITChipBitStream>> aBitStreamVector =
      make_unique<edm::DetSetVector<Phase2ITChipBitStream>>();

  auto const& tTopo = iSetup.getData(tTopoToken_);
  auto const& cablingMap = iSetup.getData(cablingMapToken_);

  auto pixelDigiHandle = iEvent.getHandle(pixelDigiToken_);
  const edm::DetSetVector<PixelDigi>& pixelDigi = *pixelDigiHandle;
  for (const auto& theDigis : pixelDigi) {
    DetId tkId = theDigis.id;
    std::vector<Phase2ITDigiHit> hitlist;
    std::vector<int> id;
    if (tkId.subdetId() == PixelSubdetector::PixelBarrel) {
      int layerNum = tTopo.pxbLayer(tkId.rawId());
      int ladderNum = tTopo.pxbLadder(tkId.rawId());
      int moduleNum = tTopo.pxbModule(tkId.rawId());
      id = {tkId.subdetId(), layerNum, ladderNum, moduleNum};
    } else if (tkId.subdetId() == PixelSubdetector::PixelEndcap) {
      int moduleNum = tTopo.pxfModule(tkId());
      int diskNum = tTopo.pxfDisk(tkId());
      int bladeNum = tTopo.pxfBlade(tkId());
      int panelNum = tTopo.pxfPanel(tkId());
      int sideNum = tTopo.pxfSide(tkId());
      id = {tkId.subdetId(), moduleNum, diskNum, bladeNum, panelNum, sideNum};
    }

    for (const auto& digi : theDigis) {
      hitlist.emplace_back(digi.row(), digi.column(), digi.adc());
    }
    // Module_SubType from the cabling map keys the chip-index convention.
    const int subtype = static_cast<int>(cablingMap.getModuleInfo(tkId.rawId()).subtype);
    std::vector<Phase2ITChip> chips = processHits(std::move(hitlist), gapMode_, subtype, tkId.rawId());

    DetSet<Phase2ITQCore> DetSetQCores(tkId);
    DetSet<Phase2ITChipBitStream> DetSetBitStream(tkId);

    for (size_t i = 0; i < chips.size(); i++) {
      Phase2ITChip chip = chips[i];
      std::vector<Phase2ITQCore> qcores = chip.getOrganizedQCores();
      for (auto& qcore : qcores) {
        DetSetQCores.push_back(qcore);
      }
      Phase2ITChipBitStream aChipBitStream(i, chip.getChipCode(dropTot_));
      DetSetBitStream.push_back(aChipBitStream);
    }

    aBitStreamVector->insert(DetSetBitStream);
    aQCoreVector->insert(DetSetQCores);
  }
  iEvent.put(std::move(aQCoreVector));
  iEvent.put(std::move(aBitStreamVector));
}

DEFINE_FWK_MODULE(PixelToBitStreamProducer);
