#include "L1Trigger/L1TGEM/interface/ME0StubBuilder.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigi.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoChamber.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoSubfunction.h"

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
  clearanceWidth_ = ps.getParameter<int32_t>("clearanceWidth");
  numOutputs_ = ps.getParameter<int32_t>("numOutputs");
  checkIds_ = ps.getParameter<bool>("checkIds");
  edgeDistance_ = ps.getParameter<int32_t>("edgeDistance");
  numOr_ = ps.getParameter<int32_t>("numOr");
  mseThreshold_ = ps.getParameter<double>("mseThreshold");
  bendAngleCut_ = ps.getParameter<double>("bendAngleCut");
  BXWindow_ = ps.getParameter<int32_t>("BXWindow");
  enablePeaking_ = ps.getParameter<bool>("enablePeaking");
  debug_ = ps.getParameter<bool>("debug");
}
ME0StubBuilder::~ME0StubBuilder() {}

void ME0StubBuilder::fillDescription(edm::ParameterSetDescription& desc) {
  desc.add<bool>("skipCentroids", false);
  desc.add<std::vector<int32_t>>("layerThresholdPatternId", {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 4, 4, 4, 4, 4});
  desc.add<std::vector<int32_t>>("layerThresholdEta", {4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4});
  desc.add<int32_t>("maxSpan", 37);
  desc.add<int32_t>("width", 192);
  desc.add<bool>("deghostPre", true);
  desc.add<bool>("deghostPost", false);
  desc.add<int32_t>("groupWidth", 16);
  desc.add<int32_t>("ghostWidth", 1);
  desc.add<bool>("xPartitionEnabled", true);
  desc.add<bool>("enableNonPointing", false);
  desc.add<int32_t>("crossPartitionSegmentWidth", 4);
  desc.add<int32_t>("clearanceWidth", 0);
  desc.add<int32_t>("numOutputs", 4);
  desc.add<bool>("checkIds", false);
  desc.add<int32_t>("edgeDistance", 2);
  desc.add<int32_t>("numOr", 2);
  desc.add<double>("mseThreshold", 0.75);
  desc.add<double>("bendAngleCut", 1.0);
  desc.add<int32_t>("BXWindow", 3);
  desc.add<bool>("enablePeaking", true);
  desc.add<bool>("debug", false);
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
  config.clearanceWidth = clearanceWidth_;
  config.numOutputs = numOutputs_;
  config.checkIds = checkIds_;
  config.edgeDistance = edgeDistance_;
  config.numOr = numOr_;
  config.mseThreshold = mseThreshold_;
  config.bendAngleCut = bendAngleCut_;
  config.BXWindow = BXWindow_;
  config.enablePeaking = enablePeaking_;

  if (config.numOutputs <= 0) 
    throw cms::Exception("ME0StubBuilder") << "numOutputs must be greater than 0";
  if (config.numOr <= 0) 
    throw cms::Exception("ME0StubBuilder") << "numOr must be greater than 0";
  if (config.BXWindow <= 0)
    throw cms::Exception("ME0StubBuilder") << "BXWindow must be greater than 0";
  if (config.BXWindow % 2 == 0)
    throw cms::Exception("ME0StubBuilder") << "BXWindow must be odd";

  std::vector<int> BXOffsetList; // list of BX offsets to consider
  for (int bx = -2; bx <= 3; ++bx) { BXOffsetList.push_back(bx); }

  int bxMin = BXOffsetList.front() - config.BXWindow / 2;
  int bxMax = BXOffsetList.back() + config.BXWindow / 2;

  std::map<uint32_t, std::vector<std::pair<ME0ChamberData, ME0ChamberBXData> > > dataMap; // rawId -> vector of (chamberData, chamberBXData) pairs for each bxOffset
  for (auto it = padDigis->begin(); it != padDigis->end(); ++it) {
    GEMDetId gemId((*it).first);
    if (gemId.station() != 0)
      continue;

    uint32_t gemRawId = (gemId.superChamberId()).rawId();
    
    // Initialize dataMap for this chamber if not already done
    if (dataMap.find(gemRawId) == dataMap.end()) {
      for (size_t idxBX = 0; idxBX < BXOffsetList.size(); ++idxBX) {
        ME0ChamberData chamberData(8, std::vector<UInt192>(6, UInt192(0)));
        ME0ChamberBXData chamberBXData(8, std::vector<std::vector<int>>(6, std::vector<int>(192, -9999)));
        dataMap[gemRawId].push_back(std::make_pair(chamberData, chamberBXData));
      }
    }

    int layer = gemId.layer();
    int ieta = gemId.ieta();
    for (auto padDigi = ((*it).second).first; padDigi != ((*it).second).second; ++padDigi) {
      int strip = (*padDigi).pad();
      int bx = (*padDigi).bx();
      if (bx < bxMin || bx > bxMax)
        continue;
      for (size_t idxBX = 0; idxBX < BXOffsetList.size(); ++idxBX) {
        int crntBX = BXOffsetList[idxBX];
        if (bx >= crntBX - config.BXWindow / 2 && bx <= crntBX + config.BXWindow / 2) { // pulse stretch
          dataMap[gemRawId][idxBX].first.at(ieta - 1).at(layer - 1) |= (UInt192(1) << (strip));
          dataMap[gemRawId][idxBX].second.at(ieta - 1).at(layer - 1).at(strip) = bx;
        }
      }
    }
  }

  // Find stub per chamber using dataMap
  int numPartitions = (config.xPartitionEnabled) ? 15 : 8;
  for (const auto& dataPair : dataMap) {
    uint32_t rawId = dataPair.first;
    GEMDetId id(rawId);
    PeakingManager peakingManager(numPartitions, config.width);
    std::vector<ME0Stub> segListProcessed;

    for (size_t idxBX = 0; idxBX < BXOffsetList.size(); ++idxBX) {

      auto data = dataPair.second[idxBX].first;
      auto bxData = dataPair.second[idxBX].second;

      std::tuple<std::vector<ME0StubPrimitive>, Config> outputChamber = processChamber(data, bxData, config, peakingManager);
      auto& segList = std::get<0>(outputChamber);
      // auto& configOut = std::get<1>(outputChamber);

      for (ME0StubPrimitive& seg : segList) {

        if (seg.patternId() == 0) 
          continue;

        if ((seg.etaPartition() % 2) != 0)
          seg.setEtaPartition(seg.etaPartition() / 2 + 1);
        else
          seg.setEtaPartition(seg.etaPartition() / 2);

        // debug
        if (debug_) {
          std::cout << std::fixed;
          std::cout.precision(4);
          std::cout << "    Eta Partition = " << seg.etaPartition() 
                    << ", Center Strip = " << seg.strip() + seg.subStrip()
                    << ", Bending angle = " << seg.bendingAngle()
                    << ", ID = " << seg.patternId()
                    << ", Hit count = " << seg.hitCount()
                    << ", Layer count = " << seg.layerCount()
                    << ", Quality = " << seg.quality()
                    << std::endl;
        }

        ME0Stub segFinal(id,
                         seg.etaPartition(),
                         seg.strip() + seg.subStrip(),
                         seg.bendingAngle(),
                         seg.layerCount(),
                         seg.quality(),
                         seg.patternId(),
                         seg.bx());

        segListProcessed.push_back(segFinal);
      }
    }
    oc.put(id, segListProcessed.begin(), segListProcessed.end());
  }
}