/*
 * AOT wrapper interface for interacting with models compiled for different batch sizes.
 *
 * Author: Marcel Rieger, Bogdan Wiederspan
 */

#include <vector>
#include <map>

#include "PhysicsTools/TensorFlowAOT/interface/Wrapper.h"

namespace tfaot {

  int Wrapper::argCount(size_t batchSize, size_t argIndex) const {
    const auto& counts = argCounts();
    const auto it = counts.find(batchSize);
    if (it == counts.end()) {
      unknownBatchSize(batchSize, "argCount()");
    }
    if (argIndex >= it->second.size()) {
      unknownArgument(argIndex, "argCount()");
    }
    return it->second.at(argIndex);
  }

  int Wrapper::argCountNoBatch(size_t argIndex) const {
    const auto& counts = argCountsNoBatch();
    if (argIndex >= counts.size()) {
      unknownArgument(argIndex, "argCountNoBatch()");
    }
    return counts.at(argIndex);
  }

  int Wrapper::resultCount(size_t batchSize, size_t resultIndex) const {
    const auto& counts = resultCounts();
    const auto it = counts.find(batchSize);
    if (it == counts.end()) {
      unknownBatchSize(batchSize, "resultCount()");
    }
    if (resultIndex >= it->second.size()) {
      unknownResult(resultIndex, "resultCount()");
    }
    return it->second.at(resultIndex);
  }

  int Wrapper::resultCountNoBatch(size_t resultIndex) const {
    const auto& counts = resultCountsNoBatch();
    if (resultIndex >= counts.size()) {
      unknownResult(resultIndex, "resultCountNoBatch()");
    }
    return counts[resultIndex];
  }

  void Wrapper::run(size_t batchSize) {
    if (!runSilent(batchSize)) {
      throw cms::Exception("FailedRun") << "evaluation with batch size " << batchSize << " failed for model '" << name_;
    }
  }

}  // namespace tfaot
