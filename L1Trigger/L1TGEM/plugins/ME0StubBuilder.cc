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
  skip_centroids = ps.getParameter<bool>("skip_centroids");
  ly_thresh_patid = ps.getParameter<std::vector<int32_t>>("ly_thresh_patid");
  ly_thresh_eta = ps.getParameter<std::vector<int32_t>>("ly_thresh_eta");
  max_span = ps.getParameter<int32_t>("max_span");
  width = ps.getParameter<int32_t>("width");
  deghost_pre = ps.getParameter<bool>("deghost_pre");
  deghost_post = ps.getParameter<bool>("deghost_post");
  group_width = ps.getParameter<int32_t>("group_width");
  ghost_width = ps.getParameter<int32_t>("ghost_width");
  x_prt_en = ps.getParameter<bool>("x_prt_en");
  en_non_pointing = ps.getParameter<bool>("en_non_pointing");
  cross_part_seg_width = ps.getParameter<int32_t>("cross_part_seg_width");
  num_outputs = ps.getParameter<int32_t>("num_outputs");
  check_ids = ps.getParameter<bool>("check_ids");
  edge_distance = ps.getParameter<int32_t>("edge_distance");
  num_or = ps.getParameter<int32_t>("num_or");
  mse_thresh = ps.getParameter<double>("mse_thresh");
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

void ME0StubBuilder::build(const GEMPadDigiCollection* paddigis, ME0StubCollection& oc) {
  Config config;
  config.skip_centroids = skip_centroids;
  config.ly_thresh_patid = ly_thresh_patid;
  config.ly_thresh_eta = ly_thresh_eta;
  config.max_span = max_span;
  config.width = width;
  config.deghost_pre = deghost_pre;
  config.deghost_post = deghost_post;
  config.group_width = group_width;
  config.ghost_width = ghost_width;
  config.x_prt_en = x_prt_en;
  config.en_non_pointing = en_non_pointing;
  config.cross_part_seg_width = cross_part_seg_width;
  config.num_outputs = num_outputs;
  config.check_ids = check_ids;
  config.edge_distance = edge_distance;
  config.num_or = num_or;

  std::map<uint32_t, std::pair<ME0ChamberData, ME0ChamberBXData>> DataMap;
  for (auto it = paddigis->begin(); it != paddigis->end(); ++it) {
    GEMDetId gemid((*it).first);
    if (gemid.station() != 0)
      continue;

    uint32_t gemRawId = (gemid.superChamberId()).rawId();

    if (DataMap[gemRawId].first.empty() || DataMap[gemRawId].second.empty()) {
      DataMap[gemRawId].first = std::vector<std::vector<UInt192>>(8, std::vector<UInt192>(6, UInt192(0)));
      DataMap[gemRawId].second =
          std::vector<std::vector<std::vector<int>>>(8, std::vector<std::vector<int>>(6, std::vector<int>(192, -9999)));
    }
    int layer = gemid.layer();
    int ieta = gemid.ieta();
    for (auto paddigi = ((*it).second).first; paddigi != ((*it).second).second; ++paddigi) {
      int strip = (*paddigi).pad();
      (DataMap[gemRawId].first.at(ieta - 1)).at(layer - 1) |= (UInt192(1) << (strip));
      ((DataMap[gemRawId].second.at(ieta - 1)).at(layer - 1)).at(strip) = (*paddigi).bx();
    }
  }

  // Find stub per chamber using DataMap
  for (const auto& data_pair : DataMap) {
    uint32_t rawId = data_pair.first;
    auto data = data_pair.second.first;
    auto bx_data = data_pair.second.second;

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

    std::vector<ME0StubPrimitive> SegList = process_chamber(data, bx_data, config);

    std::vector<ME0Stub> SegList_processed;

    for (ME0StubPrimitive& seg : SegList) {
      seg.fit(config.max_span);
      if (seg.MSE() >= mse_thresh) {
        seg.reset();
      }

      if (seg.PatternId() == 0)
        continue;
      if ((seg.EtaPartition() % 2) != 0)
        seg.SetEtaPartition(seg.EtaPartition() / 2 + 1);
      else
        seg.SetEtaPartition(seg.EtaPartition() / 2);

      ME0Stub seg_final(id, seg);

      SegList_processed.push_back(seg_final);
    }

    oc.put(id, SegList_processed.begin(), SegList_processed.end());
  }
}