//FIXME move it to a better place...

#ifndef RecoPixelVertexing_PixelTriplets_plugins_RecHitsMap_h
#define RecoPixelVertexing_PixelTriplets_plugins_RecHitsMap_h

#include <cstdint>
#include <unordered_map>

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// store T for each cluster
template <typename T>
class RecHitsMap {
public:
  explicit RecHitsMap(T const& d = T()) : dummy(d) {}

  void clear() { m_map.clear(); }

  void error(const GeomDetUnit& gd) const { edm::LogError("RecHitMap") << "hit not found in det " << gd.index(); }
  void error(uint32_t ind) const { edm::LogError("RecHitMap") << "hit not found in det " << ind; }

  // does not work for matched hits... (easy to extend)
  void add(TrackingRecHit const& hit, T const& v) {
    auto const& thit = static_cast<BaseTrackerRecHit const&>(hit);
    auto const& clus = thit.firstClusterRef();

    if (clus.isPixel())
      add(clus.pixelCluster(), *thit.detUnit(), v);
    else
      add(clus.stripCluster(), *thit.detUnit(), v);
  }

  template <typename Cluster>
  void add(const Cluster& cluster, const GeomDetUnit& gd, T const& v) {
    m_map[encode(cluster, gd)] = v;
  }

  template <typename Cluster>
  T const& get(const Cluster& cluster, const GeomDetUnit& gd) const {
    auto p = m_map.find(encode(cluster, gd));
    if (p != m_map.end()) {
      return (*p).second;
    }
    error(gd);
    return dummy;
  }

  T const& get(uint32_t ind, uint16_t mr, uint16_t mc) const {
    auto p = m_map.find(encode(ind, mr, mc));
    if (p != m_map.end()) {
      return (*p).second;
    }
    error(ind);
    return dummy;
  }

  static uint64_t encode(uint32_t ind, uint16_t mr, uint16_t mc) {
    uint64_t u1 = ind;
    uint64_t u2 = mr;
    uint64_t u3 = mc;
    return (u1 << 32) | (u2 << 16) | u3;
  }

  static uint64_t encode(const SiPixelCluster& cluster, const GeomDetUnit& det) {
    uint64_t u1 = det.index();
    uint64_t u2 = cluster.minPixelRow();
    uint64_t u3 = cluster.minPixelCol();
    return (u1 << 32) | (u2 << 16) | u3;
  }
  static uint64_t encode(const SiStripCluster& cluster, const GeomDetUnit& det) {
    uint64_t u1 = det.index();
    uint64_t u2 = cluster.firstStrip();
    return (u1 << 32) | u2;
  }

  std::unordered_map<uint64_t, T> m_map;
  T dummy;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_RecHitsMap_h
