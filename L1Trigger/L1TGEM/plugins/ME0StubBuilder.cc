#include "L1Trigger/L1TGEM/plugins/ME0StubBuilder.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigi.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoChamber.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace l1t::me0;

typedef std::vector<std::vector<UInt192>> ME0ChamberData;
typedef std::vector<std::vector<std::vector<int>>> ME0ChamberBXData;

ME0StubBuilder::ME0StubBuilder(const edm::ParameterSet& ps) {
  skipCentroids_ = ps.getParameter<bool>("skipCentroids");
  layerThresholdPatternId_ = ps.getParameter<std::vector<int32_t>>("layerThresholdPatternId");
  layerThresholdEta_ = ps.getParameter<std::vector<int32_t>>("layerThresholdEta");
  maxSpan_ = ps.getParameter<int32_t>("maxSpan");
  width_ = ps.getParameter<int32_t>("width");
  deghostPre_ = ps.getParameter<bool>("deghostPre");
  deghostPost_ = ps.getParameter<bool>("deghostPost");
  groupWidth_ = ps.getParameter<int32_t>("groupWidth");
  ghostWidth_ = ps.getParameter<int32_t>("ghostWidth");
  xPartitionEnabled_ = ps.getParameter<bool>("xPartitionEnabled");
  enableNonPointing_ = ps.getParameter<bool>("enableNonPointing");
  crossPartitionSegmentWidth_ = ps.getParameter<int32_t>("crossPartitionSegmentWidth");
  numOutputs_ = ps.getParameter<int32_t>("numOutputs");
  checkIds_ = ps.getParameter<bool>("checkIds");
  edgeDistance_ = ps.getParameter<int32_t>("edgeDistance");
  numOr_ = ps.getParameter<int32_t>("numOr");
  mseThreshold_ = ps.getParameter<double>("mseThreshold");
}
ME0StubBuilder::~ME0StubBuilder() {}

void ME0StubBuilder::fillDescription(edm::ParameterSetDescription& desc) {
  desc.add<bool>("skip_centroids", false);
  desc.add<std::vector<int32_t>>("ly_thresh_patid", {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 4, 4, 4, 4, 4});
  desc.add<std::vector<int32_t>>("ly_thresh_eta", {4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4});
  desc.add<int32_t>("max_span", 37);
  desc.add<int32_t>("width", 192);
  desc.add<bool>("deghost_pre", true);
  desc.add<bool>("deghost_post", true);
  desc.add<int32_t>("group_width", 8);
  desc.add<int32_t>("ghost_width", 1);
  desc.add<bool>("x_prt_en", true);
  desc.add<bool>("en_non_pointing", false);
  desc.add<int32_t>("cross_part_seg_width", 4);
  desc.add<int32_t>("num_outputs", 4);
  desc.add<bool>("check_ids", false);
  desc.add<int32_t>("edge_distance", 2);
  desc.add<int32_t>("num_or", 2);
  desc.add<double>("mse_thresh", 0.75);
}

void ME0StubBuilder::build(const GEMPadDigiCollection* padDigis, ME0StubCollection& oc) {
  Config config;
  config.skipCentroids = skipCentroids_;
  config.layerThresholdPatternId = layerThresholdPatternId_;
  config.layerThresholdEta = layerThresholdEta_;
  config.maxSpan = maxSpan_;
  config.width = width_;
  config.deghostPre = deghostPre_;
  config.deghostPost = deghostPost_;
  config.groupWidth = groupWidth_;
  config.ghostWidth = ghostWidth_;
  config.xPartitionEnabled = xPartitionEnabled_;
  config.enableNonPointing = enableNonPointing_;
  config.crossPartitionSegmentWidth = crossPartitionSegmentWidth_;
  config.numOutputs = numOutputs_;
  config.checkIds = checkIds_;
  config.edgeDistance = edgeDistance_;
  config.numOr = numOr_;

  std::map<uint32_t, std::pair<ME0ChamberData, ME0ChamberBXData>> dataMap;
  for (auto it = padDigis->begin(); it != padDigis->end(); ++it) {
    GEMDetId gemId((*it).first);
    if (gemId.station() != 0)
      continue;

    uint32_t gemRawId = (gemId.superChamberId()).rawId();

    if (dataMap[gemRawId].first.empty() || dataMap[gemRawId].second.empty()) {
      dataMap[gemRawId].first = std::vector<std::vector<UInt192>>(8, std::vector<UInt192>(6, UInt192(0)));
      dataMap[gemRawId].second =
          std::vector<std::vector<std::vector<int>>>(8, std::vector<std::vector<int>>(6, std::vector<int>(192, -9999)));
    }
    int layer = gemId.layer();
    int ieta = gemId.ieta();
    for (auto padDigi = ((*it).second).first; padDigi != ((*it).second).second; ++padDigi) {
      int strip = (*padDigi).pad();
      (dataMap[gemRawId].first.at(ieta - 1)).at(layer - 1) |= (UInt192(1) << (strip));
      ((dataMap[gemRawId].second.at(ieta - 1)).at(layer - 1)).at(strip) = (*padDigi).bx();
    }
  }

  // Find stub per chamber using dataMap
  for (const auto& dataPair : dataMap) {
    uint32_t rawId = dataPair.first;
    auto data = dataPair.second.first;
    auto bxData = dataPair.second.second;

    bool isNoneZero = false;
    for (const auto& etaData : data) {
      for (const auto& ly : etaData) {
        if (ly.any()) {
          isNoneZero = true;
          break;
        }
      }
      if (isNoneZero)
        break;
    }
    if (!isNoneZero)
      continue;

    GEMDetId id(rawId);

    std::vector<ME0StubPrimitive> segList = processChamber(data, bxData, config);

    std::vector<ME0Stub> segListProcessed;

    for (ME0StubPrimitive& seg : segList) {
      seg.fit(config.maxSpan);
      if (seg.mse() >= mseThreshold_) {
        seg.reset();
      }

      if (seg.patternId() == 0)
        continue;
      if ((seg.etaPartition() % 2) != 0)
        seg.setEtaPartition(seg.etaPartition() / 2 + 1);
      else
        seg.setEtaPartition(seg.etaPartition() / 2);

      ME0Stub segFinal(id, seg);

      segListProcessed.push_back(segFinal);
    }

    oc.put(id, segListProcessed.begin(), segListProcessed.end());
  }
}