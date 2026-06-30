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
#include <optional>
#include "L1Trigger/L1TGEM/interface/ME0StubPrimitive.h"
#include "L1Trigger/L1TGEM/interface/ME0StubAlgoMask.h"

namespace l1t {
  namespace me0 {
    typedef std::bitset<192> UInt192;

    class PeakingManager {
    private:
      std::vector<std::vector<std::vector<ME0StubPrimitive>>>
          segs_;                                // [oldest / old][partition][sbit] : size = (2, 15, 192)
      std::vector<std::vector<bool>> trigger_;  // [partition][sbit]
    public:
      PeakingManager() = default;
      PeakingManager(const int numPart = 15, const int width = 192) {
        segs_ = std::vector<std::vector<std::vector<ME0StubPrimitive>>>(
            2, std::vector<std::vector<ME0StubPrimitive>>(numPart, std::vector<ME0StubPrimitive>(width)));
        trigger_ = std::vector<std::vector<bool>>(numPart, std::vector<bool>(width, false));
      }
      std::vector<ME0StubPrimitive> processSegments(const int partition, const std::vector<ME0StubPrimitive>& newSegs);
      std::vector<std::vector<bool>> getTrigger() const { return trigger_; };
      std::vector<std::vector<std::vector<ME0StubPrimitive>>> getSegs() const { return segs_; };
    };

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
      int32_t clearanceWidth;
      bool xPartitionEnabled;
      bool enableNonPointing;
      int32_t crossPartitionSegmentWidth;
      int32_t numOutputs;
      bool checkIds;
      int32_t edgeDistance;
      int32_t numOr;
      double mseThreshold;
      double bendAngleCut;
      int32_t BXWindow;
      int32_t pulseStretchBx;
      bool peakingEnabled;
      // PeakingManager peakingManager;
      // void initializePeakingManager();
    };

    int countOnes(uint64_t x);
    int maxClusterSize(uint64_t x);
    UInt192 setBit(int index, UInt192 num);
    UInt192 clearBit(int index, UInt192 num);
    uint64_t oneBitMask(int num);
    std::vector<int> findOnes(uint64_t& data);
    std::pair<int, std::vector<int>> findCentroid(uint64_t& data);
    std::vector<std::vector<ME0StubPrimitive>> chunk(const std::vector<ME0StubPrimitive>& inList, int n);
    void segmentSorter(std::vector<ME0StubPrimitive>& segments, int n);
    std::vector<int> concatVector(const std::vector<std::vector<int>>& vec);
    std::vector<ME0StubPrimitive> concatVector(const std::vector<std::vector<ME0StubPrimitive>>& vec);
  }  // namespace me0
}  // namespace l1t
#endif