#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPatUnitMux.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

uint64_t l1t::me0::parseData(const l1t::me0::UInt192& data, int strip, int maxSpan) {
  l1t::me0::UInt192 dataShifted;
  uint64_t parsedData;
  if (strip < maxSpan / 2 + 1) {
    dataShifted = data << (maxSpan / 2 - strip);
    parsedData = (dataShifted & l1t::me0::UInt192(0xffffffffffffffff >> (64 - maxSpan))).to_ullong();
  } else {
    dataShifted = data >> (strip - maxSpan / 2);
    parsedData = (dataShifted & l1t::me0::UInt192(0xffffffffffffffff >> (64 - maxSpan))).to_ullong();
  }
  return parsedData;
}
std::vector<uint64_t> l1t::me0::extractDataWindow(const std::vector<l1t::me0::UInt192>& prtData,
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
std::vector<ME0StubPrimitive> l1t::me0::patMux(const std::vector<l1t::me0::UInt192>& partitionData,
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

  std::vector<ME0StubPrimitive> outSegs;
  if (config.enablePeaking) {
    outSegs = peakingManager.processSegments(partition, newSegs);
  } else {
    outSegs = newSegs;
  }

  if (debug) {
    auto seg_print = outSegs;
    int num = 0;
    LogTrace("ME0StubAlgoPatUnitMux") << "Partition = " << partition << "\n";
    for (const auto& seg : seg_print) {
      LogTrace("ME0StubAlgoPatUnitMux") << seg.quality() << "/" << seg.strip() << "/" << seg.patternId() << " ";
      num++;
      if (num % 10 == 0) {
        LogTrace("ME0StubAlgoPatUnitMux") << "\n";
      }
    }
    LogTrace("ME0StubAlgoPatUnitMux") << "\n";
  }

  return outSegs;
}
