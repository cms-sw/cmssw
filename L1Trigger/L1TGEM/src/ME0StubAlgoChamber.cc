#include "L1Trigger/L1TGEM/interface/ME0StubAlgoChamber.h"
#include <algorithm>

using namespace l1t::me0;

std::vector<std::vector<ME0StubPrimitive>> l1t::me0::crossPartitionCancellation(
    std::vector<std::vector<ME0StubPrimitive>>& segments, int crossPartSegWidth) {
  ME0StubPrimitive seg;
  ME0StubPrimitive seg1;
  ME0StubPrimitive seg2;

  int strip;
  int seg1MaxQuality;
  int seg2MaxQuality;
  int seg1MaxQualityIndex;
  int seg2MaxQualityIndex;

  for (int i = 1; i < static_cast<int>(segments.size()); i += 2) {
    for (int l = 0; l < static_cast<int>(segments[i].size()); ++l) {
      seg = segments[i][l];
      if (seg.patternId() == 0)
        continue;

      strip = seg.strip();
      seg1MaxQuality = -9999;
      seg2MaxQuality = -9999;
      seg1MaxQualityIndex = -9999;
      seg2MaxQualityIndex = -9999;

      for (int j = 0; j < static_cast<int>(segments[i - 1].size()); ++j) {
        seg1 = segments[i - 1][j];
        if (seg1.patternId() == 0)
          continue;
        if (std::abs(strip - seg1.strip()) <= crossPartSegWidth) {
          if (seg1.quality() > seg1MaxQuality) {
            if (seg1MaxQualityIndex != -9999)
              (segments[i - 1][seg1MaxQualityIndex]).reset();
            seg1MaxQualityIndex = j;
            seg1MaxQuality = seg1.quality();
          }
        }
      }
      for (int k = 0; k < static_cast<int>(segments[i + 1].size()); ++k) {
        seg2 = segments[i + 1][k];
        if (seg2.patternId() == 0)
          continue;
        if (std::abs(strip - seg2.strip()) <= crossPartSegWidth) {
          if (seg2.quality() > seg2MaxQuality) {
            if (seg2MaxQualityIndex != -9999)
              (segments[i + 1][seg2MaxQualityIndex]).reset();
            seg2MaxQualityIndex = k;
            seg2MaxQuality = seg2.quality();
          }
        }
      }

      if ((seg1MaxQualityIndex != -9999) && (seg2MaxQualityIndex != -9999)) {
        segments[i - 1][seg1MaxQualityIndex].reset();
        segments[i + 1][seg2MaxQualityIndex].reset();
      } else if (seg1MaxQualityIndex != -9999) {
        segments[i][l].reset();
      } else if (seg2MaxQualityIndex != -9999) {
        segments[i][l].reset();
      }
    }
  }
  return segments;
}

std::vector<ME0StubPrimitive> l1t::me0::processChamber(const std::vector<std::vector<UInt192>>& chamberData,
                                                       const std::vector<std::vector<std::vector<int>>>& chamberBxData,
                                                       Config& config) {
  std::vector<std::vector<ME0StubPrimitive>> segments;
  int numFinder = (config.xPartitionEnabled) ? 15 : 8;

  std::vector<std::vector<UInt192>> data(numFinder, std::vector<UInt192>(6, UInt192(0)));
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
    const std::vector<UInt192>& partitionData = data[partition];
    const std::vector<std::vector<int>>& partitionBxData = bxData[partition];
    const std::vector<ME0StubPrimitive>& segs = processPartition(partitionData, partitionBxData, partition, config);
    segments.push_back(segs);
  }

  if (config.crossPartitionSegmentWidth > 0) {
    segments = crossPartitionCancellation(segments, config.crossPartitionSegmentWidth);
  }

  // pick the best N outputs from each partition
  for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
    segmentSorter(segments[i], config.numOutputs);
  }

  // join each 2 partitions and pick the best N outputs from them
  std::vector<std::vector<ME0StubPrimitive>> joinedSegments;
  for (int i = 1; i < static_cast<int>(segments.size()); i += 2) {
    std::vector<std::vector<ME0StubPrimitive>> seed = {segments[i - 1], segments[i]};
    std::vector<ME0StubPrimitive> pair = concatVector(seed);
    joinedSegments.push_back(pair);
  }
  joinedSegments.push_back(segments[14]);
  for (int i = 0; i < static_cast<int>(joinedSegments.size()); ++i) {
    segmentSorter(joinedSegments[i], config.numOutputs);
  }

  // concatenate together all of the segments, sort them, and pick the best N outputs
  std::vector<ME0StubPrimitive> concatenated = concatVector(joinedSegments);
  segmentSorter(concatenated, config.numOutputs);

  return concatenated;
}
