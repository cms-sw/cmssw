/*
 * AOT batching rules and strategies.
 *
 * Author: Marcel Rieger, Bogdan Wiederspan
 */

#include <ostream>
#include <algorithm>

#include "PhysicsTools/TensorFlowAOT/interface/Batching.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace tfaot {

  BatchRule::BatchRule(size_t batchSize, const std::vector<size_t>& sizes, size_t lastPadding)
      : batchSize_(batchSize), sizes_(sizes), lastPadding_(lastPadding) {
    // sizes must not be empty
    if (sizes.size() == 0) {
      throw cms::Exception("EmptySizes") << "no batch sizes provided for stitching";
    }

    // the padding must be smaller than the last size
    size_t lastSize = sizes[sizes.size() - 1];
    if (lastPadding >= lastSize) {
      throw cms::Exception("WrongPadding")
          << "padding " << lastPadding << " must be smaller than last size " << lastSize;
    }

    // compute the covered batch size
    size_t sizeSum = 0;
    for (const size_t& s : sizes_) {
      sizeSum += s;
    }
    if (lastPadding > sizeSum) {
      throw cms::Exception("WrongPadding")
          << "padding " << lastPadding << " must not be larger than sum of sizes " << sizeSum;
    }
    sizeSum -= lastPadding;

    // compare to given batch size
    if (batchSize != sizeSum) {
      throw cms::Exception("WrongBatchSize")
          << "batch size " << batchSize << " does not match sum of sizes - padding " << sizeSum;
    }
  }

  const BatchRule& BatchStrategy::getRule(size_t batchSize) const {
    const auto it = rules_.find(batchSize);
    if (it == rules_.end()) {
      throw cms::Exception("UnknownBatchSize") << "batchSize " << batchSize << " not known to batching strategy";
    }
    return it->second;
  }

  std::ostream& operator<<(std::ostream& out, const BatchRule& rule) {
    out << "BatchRule(batchSize=" << rule.getBatchSize() << ", sizes=";
    for (size_t i = 0; i < rule.nSizes(); i++) {
      out << (i == 0 ? "" : ",") << rule.getSizes()[i];
    }
    return out << ", lastPadding=" << rule.getLastPadding() + ")";
  }

  void BatchStrategy::setDefaultRule(size_t batchSize, const std::vector<size_t>& availableBatchSizes) {
    std::vector<size_t> sizes;
    size_t lastPadding = 0;

    // many implementations are possible here, but for simplicity assume the most simple one:
    // if there is an exact match, use it, and otherwise repeat the smallest available size
    // n times and potentially add padding
    if (std::find(availableBatchSizes.begin(), availableBatchSizes.end(), batchSize) != availableBatchSizes.end()) {
      sizes.push_back(batchSize);
    } else {
      size_t smallestBatchSize = *std::min_element(availableBatchSizes.begin(), availableBatchSizes.end());
      size_t rest = batchSize % smallestBatchSize;
      size_t n = (batchSize / smallestBatchSize) + (rest == 0 ? 0 : 1);
      lastPadding = rest == 0 ? 0 : (smallestBatchSize - rest);
      sizes.resize(n, smallestBatchSize);
    }

    // create and register the rule
    setRule(BatchRule(batchSize, sizes, lastPadding));
  }

}  // namespace tfaot
