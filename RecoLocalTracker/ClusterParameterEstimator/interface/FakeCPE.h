#ifndef RecoLocalTracker_Cluster_Parameter_Estimator_Fake_H
#define RecoLocalTracker_Cluster_Parameter_Estimator_Fake_H

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"


#include <cstdint>
#include <unordered_map>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class FakeCPE {
public:

  using LocalValues = std::pair<LocalPoint,LocalError>;

  // store position and error for each cluster...
  class Map {
  public: 
     using LocalValues = std::pair<LocalPoint,LocalError>;
     void clear() {m_map.clear();}
     void error(const GeomDetUnit& gd) const {edm::LogError("FakeCPE") << "hit not found in det " << gd.geographicalId().rawId();  }
     template<typename Cluster>
     void add(const Cluster& cluster, const GeomDetUnit& gd,LocalValues const & lv) { m_map[encode(cluster,gd)] = lv; }

     template<typename Cluster>
     LocalValues const & get(const Cluster& cluster, const GeomDetUnit& gd) const {
       auto p = m_map.find(encode(cluster,gd));
       if (p!=m_map.end()) { return (*p).second; }
       error(gd);
       return dummy;
     }

     static uint64_t encode(const SiPixelCluster& cluster, const GeomDetUnit& det) {
          uint64_t u1 = det.geographicalId().rawId();
          uint64_t u2 = cluster.minPixelRow();
          uint64_t u3 = cluster.minPixelCol();
          return (u1<<32) | (u2<<16) | u3;
     }
     static uint64_t encode(const SiStripCluster& cluster, const GeomDetUnit& det) {
          uint64_t u1 = det.geographicalId().rawId();
          uint64_t u2 = cluster.firstStrip();
       	  return (u1<<32) | u2;
     }

  private: 
     std::unordered_map<uint64_t,LocalValues> m_map;
     LocalValues dummy;
  };
 

  Map & map() { return m_map;}
  Map const & map() const { return m_map;}

private:

  Map m_map;

};

#endif
