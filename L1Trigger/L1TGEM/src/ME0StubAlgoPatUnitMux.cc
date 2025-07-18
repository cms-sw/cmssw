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
std::vector<uint64_t> l1t::me0::extractDataWindow(const std::vector<UInt192>& layerData, int strip, int maxSpan) {
  std::vector<uint64_t> out;
  out.reserve(layerData.size());
  for (const UInt192& data : layerData) {
    out.push_back(parseData(data, strip, maxSpan));
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
std::vector<std::vector<int>> l1t::me0::extractBxDataWindow(const std::vector<std::vector<int>>& layerData,
                                                            int strip,
                                                            int maxSpan) {
  std::vector<std::vector<int>> out;
  out.reserve(layerData.size());
  for (const std::vector<int>& data : layerData) {
    out.push_back(parseBxData(data, strip, maxSpan));
  }
  return out;
}
std::vector<ME0StubPrimitive> l1t::me0::patMux(const std::vector<UInt192>& partitionData,
                                               const std::vector<std::vector<int>>& partitionBxData,
                                               int partition,
                                               Config& config) {
  std::vector<ME0StubPrimitive> out;
  for (int strip = 0; strip < config.width; ++strip) {
    const std::vector<uint64_t>& dataWindow = extractDataWindow(partitionData, strip, config.maxSpan);
    const std::vector<std::vector<int>>& bxDataWindow = extractBxDataWindow(partitionBxData, strip, config.maxSpan);
    const ME0StubPrimitive& seg = patUnit(dataWindow,
                                          bxDataWindow,
                                          strip,
                                          partition,
                                          config.layerThresholdPatternId,
                                          config.layerThresholdEta,
                                          config.maxSpan,
                                          config.skipCentroids,
                                          config.numOr);
    out.push_back(seg);
  }
  return out;
}
