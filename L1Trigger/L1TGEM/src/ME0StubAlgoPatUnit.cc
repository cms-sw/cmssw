#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPatUnit.h"

using namespace l1t::me0;

std::vector<uint64_t> l1t::me0::maskLayerData(const std::vector<uint64_t>& data, const Mask& mask) {
  std::vector<uint64_t> out;
  out.reserve(static_cast<int>(data.size()));
  for (int i = 0; i < static_cast<int>(data.size()); ++i) {
    out.push_back(data[i] & mask.mask[i]);
  }
  return out;
}

std::pair<std::vector<int>, double> l1t::me0::calculateCentroids(const std::vector<uint64_t>& maskedData,
                                                                 const std::vector<std::vector<int>>& partitionBxData) {
  std::vector<int> centroids;
  std::vector<int> bxs;
  for (int ly = 0; ly < static_cast<int>(maskedData.size()); ++ly) {
    auto data = maskedData[ly];
    auto bxData = partitionBxData[ly];
    const auto temp = findCentroid(data);
    int curCentroid = temp.first;
    std::vector<int> hitsIndices = temp.second;
    centroids.push_back(curCentroid);

    for (int hitIdx : hitsIndices) {
      bxs.push_back(bxData[hitIdx - 1]);
    }
  }
  if (static_cast<int>(bxs.size()) == 0) {
    return {centroids, -9999};
  }
  double bxSum = std::accumulate(bxs.begin(), bxs.end(), 0.0);
  double count = bxs.size();
  return {centroids, bxSum / count};
}

int l1t::me0::calculateHitCount(const std::vector<uint64_t>& maskedData, bool light) {
  int totHitCount = 0;
  if (light) {
    for (int ly : {0, 5}) {
      int hitLy = countOnes(maskedData[ly]);
      totHitCount += (hitLy < 7) ? hitLy : 7;
    }
  } else {
    for (uint64_t d : maskedData) {
      totHitCount += countOnes(d);
    }
  }
  return totHitCount;
}

int l1t::me0::calculateLayerCount(const std::vector<uint64_t>& maskedData) {
  int lyCount = 0;
  bool notZero;
  for (uint64_t d : maskedData) {
    notZero = (d != 0);
    lyCount += static_cast<int>(notZero);
  }
  return lyCount;
}

std::vector<int> l1t::me0::calculateClusterSize(const std::vector<uint64_t>& data) {
  std::vector<int> clusterSizePerLayer;
  clusterSizePerLayer.reserve(data.size());
  for (uint64_t x : data) {
    clusterSizePerLayer.push_back(maxClusterSize(x));
  }
  return clusterSizePerLayer;
}

std::vector<int> l1t::me0::calculateHits(const std::vector<uint64_t>& data) {
  std::vector<int> nHitsPerLayer;
  nHitsPerLayer.reserve(data.size());
  for (uint64_t x : data) {
    nHitsPerLayer.push_back(countOnes(x));
  }
  return nHitsPerLayer;
}

ME0StubPrimitive l1t::me0::patUnit(const std::vector<uint64_t>& data,
                                   const std::vector<std::vector<int>>& bxData,
                                   int strip,
                                   int partition,
                                   const std::vector<int>& lyThreshPatid,
                                   const std::vector<int>& lyThreshEta,
                                   int inputMaxSpan,
                                   bool skipCentroids,
                                   int numOr,
                                   bool lightHitCount,
                                   bool verbose) {
  // construct the dynamic_patlist (we do not use default PATLIST anymore)
  // for robustness concern, other codes might use PATLIST, so we kept the default PATLIST in subfunc
  // however, this could cause inconsistent issue, becareful! OR find a way to modify PATLIST

  /*
    takes in sample data for each layer and returns best segment

    processing pipeline is

    (1) take in 6 layers of raw data
    (2) for the X (~16) patterns available, AND together the raw data with the respective pattern masks
    (3) count the # of hits in each pattern
    (4) calculate the centroids for each pattern
    (5) process segments
    (6) choose the max of all patterns
    (7) apply a layer threshold
    */

  // (2)
  // and the layer data with the respective layer mask to
  // determine how many hits are in each layer
  // this yields a map object that can be iterated over to get,
  //    for each of the N patterns, the masked []*6 layer data
  std::vector<std::vector<uint64_t>> maskedData;
  std::vector<int> pids;
  for (const Mask& M : kLayerMask) {
    maskedData.push_back(maskLayerData(data, M));
    pids.push_back(M.id);
  }

  // (3) count # of hits & process centroids
  std::vector<int> hcs;
  std::vector<int> lcs;
  for (size_t idxPat = 0; idxPat < maskedData.size(); ++idxPat) {
    const std::vector<uint64_t>& x = maskedData[idxPat];
    hcs.push_back(
        0);  // hit count is not used in the current quality calculation, so we set it to 0 for now - can be re-enabled if needed
    // hcs.push_back(calculateHitCount(x, lightHitCount));
    lcs.push_back(calculateLayerCount(x));
  }

  if (*std::max_element(lcs.begin(), lcs.end()) < 4) {
    return ME0StubPrimitive(0, 0, 0, strip, partition);
  }

  // (4) process segments & choose the max of all patterns
  ME0StubPrimitive best{0, 0, 0, strip, partition};
  for (int i = 0; i < static_cast<int>(hcs.size()); ++i) {
    ME0StubPrimitive seg{lcs[i], hcs[i], pids[i], strip, partition};
    seg.updateQuality();
    if (best.quality() < seg.quality()) {
      best = seg;
      best.updateQuality();
    }
  }

  // (5) apply a layer threshold
  int lyThreshFinal = (lyThreshPatid[best.patternId() - 1] > lyThreshEta[partition])
                          ? lyThreshPatid[best.patternId() - 1]
                          : lyThreshEta[partition];
  if (best.layerCount() < lyThreshFinal) {
    best.reset();
  }

  if (verbose) {
    std::cout << "maskedData : " << std::endl;
    for (auto& x : maskedData) {
      for (auto& y : x) {
        std::cout << y << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "layer counts : " << std::endl;
    for (auto& lc : lcs) {
      std::cout << lc << " ";
    }
    std::cout << std::endl;
    std::cout << "layer threshold = " << lyThreshFinal << std::endl;
    std::cout << "Best segment: " << best << std::endl;
  }

  // process centroids if a segment is found
  if (skipCentroids || best.patternId() == 0) {
    best.setCentroids({0, 0, 0, 0, 0, 0});
    best.setBx(-9999);
  } else {
    const std::vector<uint64_t>& orginalSbits = maskedData[best.patternId() - 1];
    std::vector<uint64_t> extractedSbits;
    for (int ly = 0; ly < 6; ++ly) {
      int shift = kPatOffsets[best.patternId() - 1][ly];
      uint64_t extractedSbit = (shift > 0) ? (orginalSbits[ly] << shift) : (orginalSbits[ly] >> -shift);
      extractedSbits.push_back(extractedSbit);
    }
    auto temp = calculateCentroids(extractedSbits, bxData);
    std::vector<int> curPatternCentroids = temp.first;
    double curPatternBx = temp.second;
    best.setCentroids(curPatternCentroids);
    best.setBx(curPatternBx);
  }

  /*
  // (6) remove segments with large clusters for wide segments - ONLY NEEDED FOR PU200 - NOT USED AT THE MOMENT
  std::vector<int> clusterSizeMaxLimits = {3, 6, 9, 12, 15};
  std::vector<int> nHitsMaxLimits = {3, 6, 9, 12, 15};
  std::vector<int> clusterSizeCounts = calculateClusterSize(data);
  std::vector<int> nHitsCounts = calculateHits(data);
  std::vector<int> nLayersLargeClusters = {0, 0, 0, 0, 0};
  std::vector<int> nLayersLargeHits = {0, 0, 0, 0, 0};
  for (int i = 0; i < static_cast<int>(clusterSizeCounts.size()); ++i) {
    int threshold = clusterSizeMaxLimits[i];
    for (int l : clusterSizeCounts) {
      if (l > threshold) {
        nLayersLargeClusters[i]++;
      }
    }
  }
  for (int i = 0; i < static_cast<int>(nHitsMaxLimits.size()); ++i) {
    int threshold = nHitsMaxLimits[i];
    for (int l : nHitsCounts) {
      if (l > threshold) {
        nLayersLargeHits[i]++;
      }
    }
  }

  best.setMaxClusterSize(*std::max_element(clusterSizeCounts.begin(), clusterSizeCounts.end()));
  best.setMaxNoise(*std::max_element(nHitsCounts.begin(), nHitsCounts.end()));
  */

  best.setHitCount(0);
  best.updateQuality();

  return best;
}