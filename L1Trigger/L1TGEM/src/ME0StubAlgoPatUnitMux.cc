#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPatUnitMux.h"

using namespace l1t::me0;

uint64_t l1t::me0::parseData(const UInt192& data, int strip, int maxSpan) {
  UInt192 dataShifted;
  uint64_t parsedData;
  if (strip < maxSpan / 2 + 1) {
    dataShifted = data << (maxSpan / 2 - strip);
    parsedData = (dataShifted & UInt192(0xffffffffffffffff >> (64 - maxSpan))).to_ullong();
  } else {
    dataShifted = data >> (strip - maxSpan / 2);
    parsedData = (dataShifted & UInt192(0xffffffffffffffff >> (64 - maxSpan))).to_ullong();
  }
  return parsedData;
}
std::vector<uint64_t> l1t::me0::extractDataWindow(const std::vector<UInt192>& prtData,
                                                  int strip,
                                                  const std::vector<int>& layerSpans) {
  if (*std::max_element(layerSpans.begin(), layerSpans.end()) > 64) {
    throw cms::Exception("ME0StubAlgoPatUnitMux") << "Layer span exceeds 64 bits, which is not supported.";
  }
  std::vector<uint64_t> out;
  out.reserve(prtData.size());
  for (int ly = 0; ly < static_cast<int>(prtData.size()); ++ly) {
    out.push_back(parseData(prtData[ly], strip, layerSpans[ly]));
  }
  return out;
}
std::vector<int> l1t::me0::parseBxData(const std::vector<int>& bxData, int strip, int maxSpan) {
  std::vector<int> dataShifted;
  std::vector<int> parsedBxData;
  if (strip < maxSpan / 2 + 1) {
    std::vector<std::vector<int>> seed = {std::vector<int>((maxSpan / 2 - strip), -9999), bxData};
    dataShifted = concatVector(seed);
    parsedBxData = std::vector<int>(dataShifted.begin(), dataShifted.begin() + maxSpan);
  } else {
    int shift = strip - maxSpan / 2;
    int numAppendedNeeded = shift + maxSpan - static_cast<int>(bxData.size());
    if (numAppendedNeeded > 0) {
      std::vector<std::vector<int>> seed = {bxData, std::vector<int>(numAppendedNeeded, -9999)};
      dataShifted = concatVector(seed);
    } else {
      dataShifted = bxData;
    }
    parsedBxData = std::vector<int>(dataShifted.begin() + shift, dataShifted.begin() + shift + maxSpan);
  }
  return parsedBxData;
}
std::vector<std::vector<int>> l1t::me0::extractBxDataWindow(const std::vector<std::vector<int>>& prtBxData,
                                                            int strip,
                                                            int maxSpan) {
  std::vector<std::vector<int>> out;
  out.reserve(prtBxData.size());
  for (const std::vector<int>& data : prtBxData) {
    out.push_back(parseBxData(data, strip, maxSpan));
  }
  return out;
}
std::vector<ME0StubPrimitive> l1t::me0::patMux(const std::vector<UInt192>& partitionData,
                                               const std::vector<std::vector<int>>& partitionBxData,
                                               int partition,
                                               Config& config,
                                               PeakingManager& peakingManager,
                                               bool debug) {
  std::vector<ME0StubPrimitive> newSegs;
  int maxLayerSpan = *std::max_element(kLayerSpans.begin(), kLayerSpans.end());
  for (int strip = 0; strip < config.width; ++strip) {
    const std::vector<uint64_t>& dataWindow = extractDataWindow(partitionData, strip, kLayerSpans);
    const std::vector<std::vector<int>>& bxDataWindow = extractBxDataWindow(partitionBxData, strip, maxLayerSpan);
    const ME0StubPrimitive& seg = patUnit(dataWindow,
                                          bxDataWindow,
                                          strip,
                                          partition,
                                          config.layerThresholdPatternId,
                                          config.layerThresholdEta,
                                          config.maxSpan,
                                          config.skipCentroids,
                                          config.numOr,
                                          false,
                                          false);
    newSegs.push_back(seg);
  }
  auto peakingSegs = peakingManager.processSegments(partition, newSegs);
  // auto peakingSegs = newSegs;

  // bool is_debug_seg_exist = false;
  // int debug_seg_quality = 50378603;
  // int debug_seg_strip = 118;
  // int debug_seg_id = 11;
  // bool is_debug_seg_exist_2 = false;
  // int debug_seg_quality_2 = 33625914;
  // int debug_seg_strip_2 = 115;
  // int debug_seg_id_2 = 17;
  // for (const auto& seg : peakingSegs) {
  //   if (seg.quality() == debug_seg_quality && seg.strip() == debug_seg_strip && seg.patternId() == debug_seg_id) {
  //     is_debug_seg_exist = true;
  //   }
  //   if (seg.quality() == debug_seg_quality_2 && seg.strip() == debug_seg_strip_2 && seg.patternId() == debug_seg_id_2) {
  //     is_debug_seg_exist_2 = true;
  //   }
  // }
  // if (is_debug_seg_exist) {
  //   std::cout << "Partition = " << partition << std::endl;
  //   std::cout << "Found a segment with quality " << debug_seg_quality << ", strip " << debug_seg_strip << ", and pattern ID " << debug_seg_id << " (after peakingManager)" << std::endl;
  // }
  // if (is_debug_seg_exist_2) {
  //   std::cout << "Partition = " << partition << std::endl;
  //   std::cout << "Found a segment with quality " << debug_seg_quality_2 << ", strip " << debug_seg_strip_2 << ", and pattern ID " << debug_seg_id_2 << " (after peakingManager)" << std::endl;
  // }

  if (debug) {
    // auto seg_print = newSegs;
    auto seg_print = peakingSegs;
    int num = 0;
    std::cout << "Partition = " << partition << std::endl;
    for (const auto& seg : seg_print) {
      std::cout << seg.quality() << "/" << seg.strip() << "/" << seg.patternId() << " ";
      num++;
      if (num % 10 == 0) {
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }

  return peakingSegs;
}
