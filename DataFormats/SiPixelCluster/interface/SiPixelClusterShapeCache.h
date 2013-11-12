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
  SiPixelClusterShapeData();
  ~SiPixelClusterShapeData();

  const std::vector<std::pair<int, int> >& size() const { return size_; }

  bool isStraight() const { return isStraight_; }
  bool isComplete() const { return isComplete_; }
  bool hasBigPixelsOnlyInside() const { return hasBigPixelsOnlyInside_; }

  template <typename T>
  void set(const T& data) {
    size_.clear();
    size_.reserve(data.size.size());
    std::copy(data.size.begin(), data.size.end(), std::back_inserter(size_));
    isStraight_ = data.isStraight;
    isComplete_ = data.isComplete;
    hasBigPixelsOnlyInside_ = data.hasBigPixelsOnlyInside;
  }

private:
   std::vector<std::pair<int,int> > size_;
   bool isStraight_, isComplete_, hasBigPixelsOnlyInside_;
};

class SiPixelClusterShapeCache {
public:
  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> ClusterRef;

  SiPixelClusterShapeCache() {};
  explicit SiPixelClusterShapeCache(const edm::HandleBase& handle): productId_(handle.id()) {}
  explicit SiPixelClusterShapeCache(const edm::ProductID& id): productId_(id) {}
  ~SiPixelClusterShapeCache();

  void resize(size_t size) {
    data_.resize(size);
  }

  template <typename T>
  void set(const ClusterRef& cluster, const T& data) {
    assert(productId_ == cluster.id());
    data_.at(cluster.index()).set(data);
  }

  const SiPixelClusterShapeData& get(const ClusterRef& cluster) const {
    assert(productId_ == cluster.id());
    return data_.at(cluster.index());
  }

private:
  std::vector<SiPixelClusterShapeData> data_;
  edm::ProductID productId_;
};

#endif
