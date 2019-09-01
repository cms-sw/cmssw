#ifndef RecoParticleFlow_PFProducer_CommutativePairs_h
#define RecoParticleFlow_PFProducer_CommutativePairs_h

#include <utility>
#include <vector>

/**
 * Wrapper around std::vector<std::pair<T, T>> when the order of the pair elements is not relevant.
 *
 * @tparam T the type of data stored in the pairs
 */

template <class T>
class CommutativePairs {
public:
  // Insert a new pair
  void insert(T const& a, T const& b) { pairs_.emplace_back(a, b); }

  // Check if this contains (a,b) or (b,a)
  bool contains(T const& a, T const& b) const {
    for (auto const& p : pairs_) {
      if ((a == p.first && b == p.second) || (b == p.first && a == p.second)) {
        return true;
      }
    }
    return false;
  }

  // Check if this contains (a,b) or (b,a), where b is arbitrary
  bool contains(T const& a) const {
    for (auto const& p : pairs_) {
      if (a == p.first || a == p.second) {
        return true;
      }
    }
    return false;
  }

  /// Add the pairs from another CommutativePairs to this
  void concatenate(CommutativePairs<T> const& other) {
    pairs_.insert(pairs_.end(), other.pairs_.begin(), other.pairs_.end());
  }

private:
  std::vector<std::pair<T, T>> pairs_;
};

#endif
