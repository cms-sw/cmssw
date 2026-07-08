#include "L1Trigger/L1TGEM/interface/ME0StubAlgoChamber.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

std::vector<std::vector<ME0StubPrimitive>> l1t::me0::deghostingClearance(
    std::vector<std::vector<ME0StubPrimitive>>& segments, int clearanceWidth) {
  auto segsOut = segments;
  for (int prtIdx = 0; prtIdx < static_cast<int>(segments.size()); ++prtIdx) {
    for (int segIdx = 0; segIdx < static_cast<int>(segments[prtIdx].size()); ++segIdx) {
      std::vector<int> prts = {0};
      std::vector<int> chunks = {0};

      // If virtual partition, do x-prt deghosting. If at top or bottom, don't try to look out of bounds.
      if (prtIdx % 2 == 1) {
        if (prtIdx != 0)
          prts.push_back(-1);
        if (prtIdx != static_cast<int>(segments.size()) - 1)
          prts.push_back(1);
      }
      // Don't look out of bounds
      if (segIdx != 0)
        chunks.push_back(-1);
      if (segIdx != static_cast<int>(segments[prtIdx].size()) - 1)
        chunks.push_back(1);
      // Generate all permutations of (relative_partition, relative_chunk)
      std::vector<std::pair<int, int>> relativeIndices;
      for (int prt : prts) {
        for (int chunk : chunks) {
          if (prt == 0 && chunk == 0)
            continue;  // skip self
          relativeIndices.emplace_back(prt, chunk);
        }
      }
      auto seg = segments[prtIdx][segIdx];
      for (const auto& [relativePrt, relativeChunk] : relativeIndices) {
        auto otherSeg = segments[prtIdx + relativePrt][segIdx + relativeChunk];
        if (otherSeg.patternId() != 0 && std::abs(seg.strip() - otherSeg.strip()) <= clearanceWidth) {
          if (seg.quality() > otherSeg.quality()) {
            segsOut[prtIdx + relativePrt][segIdx + relativeChunk].reset();
          } else {
            segsOut[prtIdx][segIdx].reset();
          }
        }
      }
    }
  }
  return segsOut;
}

std::vector<std::vector<ME0StubPrimitive>> l1t::me0::crossPartitionCancellation(
    std::vector<std::vector<ME0StubPrimitive>>& segments, int crossPartSegWidth) {
  std::vector<std::vector<ME0StubPrimitive>> segRealKilled = segments;
  // Step 1: Kill real segments with a better nearby virtual segment
  for (int prtIdx = 1; prtIdx < static_cast<int>(segments.size()); prtIdx += 2) {
    for (int segIdx = 0; segIdx < static_cast<int>(segments[prtIdx].size()); ++segIdx) {
      ME0StubPrimitive seg = segments[prtIdx][segIdx];
      if (seg.layerCount() == 0)
        continue;
      for (int segAboveIdx = 0; segAboveIdx < static_cast<int>(segments[prtIdx - 1].size()); ++segAboveIdx) {
        ME0StubPrimitive segAbove = segments[prtIdx - 1][segAboveIdx];
        if (segAbove.layerCount() != 0 && std::abs(seg.strip() - segAbove.strip()) <= crossPartSegWidth &&
            ((seg.layerCount() << 5) + seg.patternId() > (segAbove.layerCount() << 5) + segAbove.patternId())) {
          segRealKilled[prtIdx - 1][segAboveIdx].reset();
        }
      }
      for (int segBelowIdx = 0; segBelowIdx < static_cast<int>(segments[prtIdx + 1].size()); ++segBelowIdx) {
        ME0StubPrimitive segBelow = segments[prtIdx + 1][segBelowIdx];
        if (segBelow.layerCount() != 0 && std::abs(seg.strip() - segBelow.strip()) <= crossPartSegWidth &&
            ((seg.layerCount() << 5) + seg.patternId() > (segBelow.layerCount() << 5) + segBelow.patternId())) {
          segRealKilled[prtIdx + 1][segBelowIdx].reset();
        }
      }
    }
  }
  std::vector<std::vector<ME0StubPrimitive>> segsOut = segRealKilled;
  // Step 2: Kill virtual segments that still have a nearby real segment (i.e. virtual segments with a better nearby real segment)
  for (int prtIdx = 1; prtIdx < static_cast<int>(segRealKilled.size()); prtIdx += 2) {
    for (int segIdx = 0; segIdx < static_cast<int>(segRealKilled[prtIdx].size()); ++segIdx) {
      ME0StubPrimitive seg = segRealKilled[prtIdx][segIdx];
      if (seg.layerCount() == 0)
        continue;
      for (int segAboveIdx = 0; segAboveIdx < static_cast<int>(segRealKilled[prtIdx - 1].size()); ++segAboveIdx) {
        ME0StubPrimitive segAbove = segRealKilled[prtIdx - 1][segAboveIdx];
        if (segAbove.layerCount() != 0 && std::abs(seg.strip() - segAbove.strip()) <= crossPartSegWidth) {
          segsOut[prtIdx][segIdx].reset();
        }
      }
      for (int segBelowIdx = 0; segBelowIdx < static_cast<int>(segRealKilled[prtIdx + 1].size()); ++segBelowIdx) {
        ME0StubPrimitive segBelow = segRealKilled[prtIdx + 1][segBelowIdx];
        if (segBelow.layerCount() != 0 && std::abs(seg.strip() - segBelow.strip()) <= crossPartSegWidth) {
          segsOut[prtIdx][segIdx].reset();
        }
      }
    }
  }
  return segsOut;
}

std::vector<ME0StubPrimitive> l1t::me0::processChamber(const std::vector<std::vector<l1t::me0::UInt192>>& chamberData,
                                                       const std::vector<std::vector<std::vector<int>>>& chamberBxData,
                                                       l1t::me0::Config& config,
                                                       l1t::me0::PeakingManager& peakingManager) {
  std::vector<std::vector<ME0StubPrimitive>> segments;
  int numFinder = (config.xPartitionEnabled) ? 15 : 8;

  std::vector<std::vector<l1t::me0::UInt192>> data(numFinder, std::vector<l1t::me0::UInt192>(6, l1t::me0::UInt192(0)));
  std::vector<std::vector<std::vector<int>>> bxData(numFinder,
                                                    std::vector<std::vector<int>>(6, std::vector<int>(192, -9999)));
  if (config.xPartitionEnabled) {
    for (int finder = 0; finder < numFinder; ++finder) {
      // even finders are simple, just take the partition
      if (finder % 2 == 0) {
        data[finder] = chamberData[finder / 2];
        bxData[finder] = chamberBxData[finder / 2];
      }
      // odd finders are the OR of two adjacent partitions
      else {
        data[finder][0] = chamberData[finder / 2 + 1][0];
        data[finder][1] = chamberData[finder / 2 + 1][1];
        data[finder][2] = chamberData[finder / 2][2] | chamberData[finder / 2 + 1][2];
        data[finder][3] = chamberData[finder / 2][3] | chamberData[finder / 2 + 1][3];
        data[finder][4] = chamberData[finder / 2][4];
        data[finder][5] = chamberData[finder / 2][5];

        bxData[finder][0] = chamberBxData[finder / 2 + 1][0];
        bxData[finder][1] = chamberBxData[finder / 2 + 1][1];
        for (int i = 0; i < static_cast<int>(bxData[finder][2].size()); ++i) {
          bxData[finder][2][i] = std::max(chamberBxData[finder / 2][2][i], chamberBxData[finder / 2 + 1][2][i]);
          bxData[finder][3][i] = std::max(chamberBxData[finder / 2][3][i], chamberBxData[finder / 2 + 1][3][i]);
        }
        bxData[finder][4] = chamberBxData[finder / 2][4];
        bxData[finder][5] = chamberBxData[finder / 2][5];
      }
    }
  } else {
    data = chamberData;
  }

  for (int partition = 0; partition < static_cast<int>(data.size()); ++partition) {
    const std::vector<l1t::me0::UInt192>& partitionData = data[partition];
    const std::vector<std::vector<int>>& partitionBxData = bxData[partition];
    const std::vector<ME0StubPrimitive>& segs =
        l1t::me0::processPartition(partitionData, partitionBxData, partition, config, peakingManager);
    segments.push_back(segs);
  }

  if (config.crossPartitionSegmentWidth > 0)
    segments = l1t::me0::crossPartitionCancellation(segments, config.crossPartitionSegmentWidth);
  if (config.clearanceWidth > 0)
    segments = l1t::me0::deghostingClearance(segments, config.clearanceWidth);

  // pick the best N outputs from each partition
  for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
    l1t::me0::segmentSorter(segments[i], config.numOutputs);
  }

  // join each 2 partitions and pick the best N outputs from them
  std::vector<std::vector<ME0StubPrimitive>> joinedSegments;
  for (int i = 1; i < static_cast<int>(segments.size()); i += 2) {
    std::vector<std::vector<ME0StubPrimitive>> seed = {segments[i - 1], segments[i]};
    std::vector<ME0StubPrimitive> pair = l1t::me0::concatVector(seed);
    joinedSegments.push_back(pair);
  }
  joinedSegments.push_back(segments[14]);
  for (int i = 0; i < static_cast<int>(joinedSegments.size()); ++i) {
    l1t::me0::segmentSorter(joinedSegments[i], config.numOutputs);
  }

  // concatenate together all of the segments, sort them, and pick the best N outputs
  std::vector<ME0StubPrimitive> concatenated = l1t::me0::concatVector(joinedSegments);
  l1t::me0::segmentSorter(concatenated, config.numOutputs);

  // Fit segments and bending angle cut
  for (ME0StubPrimitive& seg : concatenated) {
    if (seg.patternId() == 0)
      continue;  // skip stubs that are not valid
    seg.fit(kPatSpans[seg.patternId() - 1]);
    if (std::abs(seg.bendingAngle()) > config.bendAngleCut) {
      seg.reset();
    }
  }

  return concatenated;
}
