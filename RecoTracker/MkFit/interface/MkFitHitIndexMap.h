#ifndef RecoTracker_MkFit_MkFitHitIndexMap_h
#define RecoTracker_MkFit_MkFitHitIndexMap_h

#include "DataFormats/Provenance/interface/ProductID.h"

#include <vector>

class TrackingRecHit;

/**
 * This class provides mappings
 * - from CMSSW(ProductID, cluster index) to mkFit(layer index, hit index)
 * - from mkFit(layer index, hit index) to pointer to CMSSW hit
 */
class MkFitHitIndexMap {
public:
  // This class holds the index and layer of a hit in the hit data
  // structure passed to mkFit
  class MkFitHit {
  public:
    MkFitHit() = default;
    explicit MkFitHit(int i, int l) : index_{i}, layer_{l} {}

    int index() const { return index_; }
    int layer() const { return layer_; }

  private:
    int index_ = -1;
    int layer_ = -1;
  };

  MkFitHitIndexMap() = default;

  /**
   * Can be used to preallocate the internal vectors for CMSSW->mkFit mapping
   */
  void resizeByClusterIndex(edm::ProductID id, size_t clusterIndex);

  /**
   * Can be used to preallocate the internal vectors for mkFit->CMSSW mapping
   *
   * \param layer           Layer index (in mkFit convention)
   * \param additionalSize  Number of additional elements to make space for
   */
  void increaseLayerSize(int layer, size_t additionalSize);

  /**
   * Inserts a new hit in the mapping
   *
   * \param id            ProductID of the cluster collection
   * \param clusterIndex  Index of the cluster in the cluster collection
   * \param hit           Index and layer of the hit in the mkFit hit data structure
   * \param hitPtr        Pointer to the TrackingRecHit
   */
  void insert(edm::ProductID id, size_t clusterIndex, MkFitHit hit, const TrackingRecHit* hitPtr);

  /// Get mkFit hit index and layer
  const MkFitHit& mkFitHit(edm::ProductID id, size_t clusterIndex) const;

  /// Get CMSSW hit pointer
  const TrackingRecHit* hitPtr(MkFitHit hit) const { return mkFitToCMSSW_.at(hit.layer()).at(hit.index()).ptr; }

  /// Get CMSSW cluster index (currently used only for debugging)
  size_t clusterIndex(MkFitHit hit) const { return mkFitToCMSSW_.at(hit.layer()).at(hit.index()).clusterIndex; }

private:
  // Helper struct to map (edm::ProductID, cluster index) to MkFitHit
  struct ClusterToMkFitHit {
    explicit ClusterToMkFitHit(edm::ProductID id) : productID(id) {}
    edm::ProductID productID;
    std::vector<MkFitHit> mkFitHits;  // indexed by cluster index
  };

  // Helper struct to map MkFitHit to (TrackingRecHit *, cluster index)
  struct CMSSWHit {
    CMSSWHit() = default;
    explicit CMSSWHit(const TrackingRecHit* p, size_t i) : ptr{p}, clusterIndex{i} {}
    const TrackingRecHit* ptr = nullptr;
    size_t clusterIndex = 0;
  };

  std::vector<ClusterToMkFitHit> cmsswToMkFit_;  // mapping from CMSSW(ProductID, cluster index) -> mkFit(index, layer)
  std::vector<std::vector<CMSSWHit> > mkFitToCMSSW_;  // reverse mapping, mkFit(layer, index) -> CMSSW hit
};

#endif
