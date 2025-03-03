#ifndef L1Trigger_L1TGEM_ME0StubAlgoSubfunction_H
#define L1Trigger_L1TGEM_ME0StubAlgoSubfunction_H

#include <cmath>
#include <vector>
#include <map>
#include <cstdint>
#include <algorithm>
#include <bitset>
#include <iostream>
#include <numeric>
#include "L1Trigger/L1TGEM/interface/ME0StubPrimitive.h"

namespace l1t {
  namespace me0 {
    typedef std::bitset<192> UInt192;

    struct Config {
      bool skipCentroids;
      std::vector<int32_t> layerThresholdPatternId;
      std::vector<int32_t> layerThresholdEta;
      int32_t maxSpan;
      int32_t width;
      bool deghostPre;
      bool deghostPost;
      int32_t groupWidth;
      int32_t ghostWidth;
      bool xPartitionEnabled;
      bool enableNonPointing;
      int32_t crossPartitionSegmentWidth;
      int32_t numOutputs;
      bool checkIds;
      int32_t edgeDistance;
      int32_t numOr;
    };

    class HiLo {
    private:
    public:
      int hi, lo;
      HiLo(int hi, int lo) : hi(hi), lo(lo) {}
    };

    class PatternDefinition {
    private:
    public:
      int id;
      std::vector<HiLo> layers;
      PatternDefinition(int id, std::vector<HiLo> layers) : id(id), layers(layers) {}
    };

    class Mask {
    private:
    public:
      int id;
      std::vector<uint64_t> mask;
      Mask(int id, std::vector<uint64_t> mask) : id(id), mask(mask) {}
      std::string toString() const;
    };

    HiLo mirrorHiLo(const HiLo& layer);
    PatternDefinition mirrorPatternDefinition(const PatternDefinition& pattern, int id);
    std::vector<HiLo> createPatternLayer(double lower, double upper);

    int countOnes(uint64_t x);
    int maxClusterSize(uint64_t x);
    UInt192 setBit(int index, UInt192 num);
    UInt192 clearBit(int index, UInt192 num);
    uint64_t oneBitMask(int num);
    std::vector<int> findOnes(uint64_t& data);
    std::pair<double, std::vector<int>> findCentroid(uint64_t& data);
    std::vector<std::vector<ME0StubPrimitive>> chunk(const std::vector<ME0StubPrimitive>& inList, int n);
    void segmentSorter(std::vector<ME0StubPrimitive>& segments, int n);
    std::vector<int> concatVector(const std::vector<std::vector<int>>& vec);
    std::vector<ME0StubPrimitive> concatVector(const std::vector<std::vector<ME0StubPrimitive>>& vec);
  }  // namespace me0
}  // namespace l1t
#endif