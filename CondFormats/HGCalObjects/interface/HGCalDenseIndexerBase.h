#ifndef CondFormats_HGCalObjects_interface_HGCalDenseIndexerBase_h
#define CondFormats_HGCalObjects_interface_HGCalDenseIndexerBase_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <array>
#include <numeric>

/**
   @short this is a simple class that takes care of building a dense index for a set of categories
   the maximum number of items expected in each category is encoded in the IndexRanges_t
   the class is templated for the number of categories to use
 */
class HGCalDenseIndexerBase {
public:
  HGCalDenseIndexerBase() : HGCalDenseIndexerBase(0) {}

  HGCalDenseIndexerBase(int n) : n_(n), maxIdx_(0), vmax_(n, 0) {}

  HGCalDenseIndexerBase(std::vector<uint32_t> const &o) : n_(o.size()) { updateRanges(o); }

  void updateRanges(std::vector<uint32_t> const &o) {
    check(o.size());
    vmax_ = o;
    maxIdx_ = std::accumulate(vmax_.begin(), vmax_.end(), 1, std::multiplies<uint32_t>());
  }

  uint32_t denseIndex(std::vector<uint32_t> v) const {
    uint32_t rtn = v[0];
    for (size_t i = 1; i < n_; i++)
      rtn = rtn * vmax_[i] + v[i];
    return rtn;
  }

  std::vector<uint32_t> unpackDenseIndex(uint32_t rtn) const {
    std::vector<uint32_t> codes(n_, 0);

    const auto rend = vmax_.rend();
    for (auto rit = vmax_.rbegin(); rit != rend; ++rit) {
      size_t i = rend - rit - 1;
      codes[i] = rtn % (*rit);
      rtn = rtn / (*rit);
    }

    return codes;
  }

  uint32_t getMaxIndex() const { return maxIdx_; }

  ~HGCalDenseIndexerBase() = default;

private:
  void check(size_t osize) const {
    if (osize != n_)
      throw cms::Exception("ValueError") << " unable to update indexer max values. Expected " << n_ << " received "
                                         << osize;
  }

  uint32_t n_;
  uint32_t maxIdx_;
  std::vector<uint32_t> vmax_;

  COND_SERIALIZABLE;
};

#endif
