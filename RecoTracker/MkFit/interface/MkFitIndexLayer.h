#ifndef RecoTracker_MkFit_MkFitIndexLayer_h
#define RecoTracker_MkFit_MkFitIndexLayer_h

#include "DataFormats/Provenance/interface/ProductID.h"

#include <vector>

class TrackingRecHit;

class MkFitIndexLayer {
public:
  struct HitInfo {
    HitInfo() : index(-1), layer(-1) {}
    HitInfo(int i, int l) : index(i), layer(l) {}
    int index;
    int layer;
  };

  struct Coll {
    explicit Coll(edm::ProductID id) : productID(id) {}
    edm::ProductID productID;
    std::vector<HitInfo> infos;  // indexed by cluster index
  };

  MkFitIndexLayer() = default;

  void insert(edm::ProductID id, size_t clusterIndex, int hit, int layer, const TrackingRecHit *hitPtr);

  const HitInfo &get(edm::ProductID id, size_t clusterIndex) const;

  const TrackingRecHit *getHitPtr(int layer, int hit) const { return hits_.at(layer).at(hit).ptr; }

  size_t getClusterIndex(int layer, int hit) const { return hits_.at(layer).at(hit).clusterIndex; }

private:
  struct CMSSWHit {
    const TrackingRecHit *ptr = nullptr;
    size_t clusterIndex = 0;
  };

  std::vector<Coll> colls_;                   // mapping from CMSSW(ProductID, index) -> mkFit(index, layer)
  std::vector<std::vector<CMSSWHit> > hits_;  // reverse mapping, mkFit(layer, index) -> CMSSW hit
};

#endif
