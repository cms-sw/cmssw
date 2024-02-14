#ifndef PHYSICSTOOLS_TENSORFLOWAOT_BATCHING_H
#define PHYSICSTOOLS_TENSORFLOWAOT_BATCHING_H

/*
 * AOT batching rules and strategies.
 *
 * Author: Marcel Rieger, Bogdan Wiederspan
 */

#include <cstddef>
#include <vector>
#include <map>
#include <ostream>

namespace tfaot {

  // rule defining how a certain batch size should be composed of various smaller sizes plus an
  // optional padding that is applied to the last size
  class BatchRule {
  public:
    // constructor
    explicit BatchRule(size_t batchSize, const std::vector<size_t>& sizes, size_t lastPadding = 0);

    // destructor
    ~BatchRule() = default;

    // getter for the batch size
    size_t getBatchSize() const { return batchSize_; }

    // getter for available sizes
    const std::vector<size_t>& getSizes() const { return sizes_; }

    // getter for the last padding value
    size_t getLastPadding() const { return lastPadding_; }

    // returns the number of available sizes
    size_t nSizes() const { return sizes_.size(); }

    // getter for the registered size at index i
    size_t getSize(size_t i) const { return sizes_[i]; }

  private:
    size_t batchSize_;
    std::vector<size_t> sizes_;
    size_t lastPadding_;
  };

  // stream operator
  std::ostream& operator<<(std::ostream& out, const BatchRule& rule);

  // the batch strategy is a collection of batch rules registered to certain batch sizes
  class BatchStrategy {
  public:
    // constructor
    explicit BatchStrategy() = default;

    // destructor
    ~BatchStrategy() = default;

    // registers a new rule for a batch size
    void setRule(const BatchRule& rule) { rules_.insert_or_assign(rule.getBatchSize(), rule); }

    // returns whether a rule was already registered for a certain batch size
    bool hasRule(size_t batchSize) const { return rules_.find(batchSize) != rules_.end(); }

    // returns a rule registered previously for a certain batch size
    const BatchRule& getRule(size_t batchSize) const;

    // registers a new rule for a certain batch size according to a certain algorithm
    void setDefaultRule(size_t batchSize, const std::vector<size_t>& availableBatchSizes);

  private:
    std::map<size_t, BatchRule> rules_;
  };

}  // namespace tfaot

#endif  // PHYSICSTOOLS_TENSORFLOWAOT_BATCHING_H
