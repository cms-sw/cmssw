#ifndef DataFormats_SiStripCluster_SiStripApproximateClusterCollection_h
#define DataFormats_SiStripCluster_SiStripApproximateClusterCollection_h

#include <vector>

#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"

/**
 * This class provides a minimal interface that resembles
 * edmNew::DetSetVector, but is crafted such that we are comfortable
 * to provide an infinite backwards compatibility guarantee for it
 * (like all RAW data). Any modifications need to be made with care.
 * Please consult core software group if in doubt.
**/
class SiStripApproximateClusterCollection {
public:
  // Helper classes to make creation and iteration easier
  class Filler {
  public:
    void push_back(SiStripApproximateCluster const& cluster) { clusters_.push_back(cluster); }

  private:
    friend SiStripApproximateClusterCollection;
    Filler(std::vector<SiStripApproximateCluster>& clusters) : clusters_(clusters) {}

    std::vector<SiStripApproximateCluster>& clusters_;
  };

  class const_iterator;
  class DetSet {
  public:
    using const_iterator = std::vector<SiStripApproximateCluster>::const_iterator;

    unsigned int id() const { return coll_->detIds_[detIndex_]; }

    const_iterator begin() const { return coll_->clusters_.begin() + clusBegin_; }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return coll_->clusters_.begin() + clusEnd_; }
    const_iterator cend() const { return end(); }

  private:
    friend SiStripApproximateClusterCollection::const_iterator;
    DetSet(SiStripApproximateClusterCollection const* coll, unsigned int detIndex)
        : coll_(coll),
          detIndex_(detIndex),
          clusBegin_(coll_->beginIndices_[detIndex]),
          clusEnd_(detIndex == coll_->beginIndices_.size() - 1 ? coll->clusters_.size()
                                                               : coll_->beginIndices_[detIndex + 1]) {}

    SiStripApproximateClusterCollection const* const coll_;
    unsigned int const detIndex_;
    unsigned int const clusBegin_;
    unsigned int const clusEnd_;
  };

  class const_iterator {
  public:
    DetSet operator*() const { return DetSet(coll_, index_); }

    const_iterator& operator++() {
      ++index_;
      if (index_ == coll_->detIds_.size()) {
        *this = const_iterator();
      }
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator clone = *this;
      ++(*this);
      return clone;
    }

    bool operator==(const_iterator const& other) const { return coll_ == other.coll_ and index_ == other.index_; }
    bool operator!=(const_iterator const& other) const { return not operator==(other); }

  private:
    friend SiStripApproximateClusterCollection;
    // default-constructed object acts as the sentinel
    const_iterator() = default;
    const_iterator(SiStripApproximateClusterCollection const* coll) : coll_(coll) {}

    SiStripApproximateClusterCollection const* coll_ = nullptr;
    unsigned int index_ = 0;
  };

  // Actual public interface
  SiStripApproximateClusterCollection() = default;

  void reserve(std::size_t dets, std::size_t clusters);
  Filler beginDet(unsigned int detId);

  const_iterator begin() const { return const_iterator(clusters_.empty() ? nullptr : this); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(); }
  const_iterator cend() const { return end(); }

private:
  // The detIds_ and beginIndices_ have one element for each Det. An
  // element of beginIndices_ points to the first cluster of the Det
  // in clusters_.
  std::vector<unsigned int> detIds_;  // DetId for the Det
  std::vector<unsigned int> beginIndices_;
  std::vector<SiStripApproximateCluster> clusters_;
};

#endif
