// -*- c++ -*-
#ifndef DataFormats_SiPixelCluster_SiPixelClusterShapeData_h
#define DataFormats_SiPixelCluster_SiPixelClusterShapeData_h

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/HandleBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <cassert>

class SiPixelClusterShapeData {
public:
  typedef std::vector<std::pair<int, int> >::const_iterator const_iterator;
  typedef std::pair<const_iterator, const_iterator> Range;
  SiPixelClusterShapeData(const_iterator begin, const_iterator end, bool isStraight, bool isComplete, bool hasBigPixelsOnlyInside):
    begin_(begin), end_(end), isStraight_(isStraight), isComplete_(isComplete), hasBigPixelsOnlyInside_(hasBigPixelsOnlyInside)
  {}
  ~SiPixelClusterShapeData();

  Range size() const { return std::make_pair(begin_, end_); }

  bool isStraight() const { return isStraight_; }
  bool isComplete() const { return isComplete_; }
  bool hasBigPixelsOnlyInside() const { return hasBigPixelsOnlyInside_; }

private:
  const_iterator begin_, end_;
  const bool isStraight_, isComplete_, hasBigPixelsOnlyInside_;
};

class SiPixelClusterShapeCache {
public:
  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> ClusterRef;

  struct Field {
    Field(unsigned i, bool s, bool c, bool h):
      straight(s), complete(c), has(h), index(i) {}
    unsigned straight:1;
    unsigned complete:1;
    unsigned has:1;
    unsigned index:29;
  };

  SiPixelClusterShapeCache() {};
  explicit SiPixelClusterShapeCache(const edm::HandleBase& handle): productId_(handle.id()) {}
  explicit SiPixelClusterShapeCache(const edm::ProductID& id): productId_(id) {}
  ~SiPixelClusterShapeCache();

  void reserve(size_t size) {
    assert(size <= 2<<29); // maximum size
    data_.reserve(size);
    sizeData_.reserve(size);
  }

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  void shrink_to_fit() {
    data_.shrink_to_fit();
    sizeData_.shrink_to_fit();
  }
#endif

  template <typename T>
  void push_back(const ClusterRef& cluster, const T& data) {
    assert(productId_ == cluster.id());
    assert(cluster.index() == data_.size()); // ensure data are pushed in correct order

    std::copy(data.size.begin(), data.size.end(), std::back_inserter(sizeData_));
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    data_.emplace_back(sizeData_.size(), data.isStraight, data.isComplete, data.hasBigPixelsOnlyInside);
#else
    data_.push_back(Field(sizeData_.size(), data.isStraight, data.isComplete, data.hasBigPixelsOnlyInside));
#endif
  }

  SiPixelClusterShapeData get(const ClusterRef& cluster) const {
    assert(productId_ == cluster.id());
    assert(cluster.index() < data_.size());
    unsigned beg = 0;
    if(cluster.index() > 0)
      beg = data_[cluster.index()-1].index;

    Field f = data_[cluster.index()];
    unsigned end = f.index;

    return SiPixelClusterShapeData(sizeData_.begin()+beg, sizeData_.begin()+end,
                                   f.straight, f.complete, f.has);
  }

private:
  std::vector<Field> data_;
  std::vector<std::pair<int, int> > sizeData_;
  edm::ProductID productId_;
};

#endif
