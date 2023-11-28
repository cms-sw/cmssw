// Author: Aurora Perego, Fabio Cossutti - aurora.perego@cern.ch, fabio.cossutti@ts.infn.it
// Date: 05/2023

#ifndef __RecoLocalFastTime_FTLCommonAlgos_MTDGeomUtil_h__
#define __RecoLocalFastTime_FTLCommonAlgos_MTDGeomUtil_h__

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace mtd {
  class MTDGeomUtil {
  public:
    MTDGeomUtil() : geom_(nullptr), topology_(nullptr) {}
    ~MTDGeomUtil() {}

    void setGeometry(MTDGeometry const* geom);
    void setTopology(MTDTopology const* topo);

    bool isETL(const DetId&) const;
    bool isBTL(const DetId&) const;

    std::pair<LocalPoint, GlobalPoint> position(const DetId& id, int row = 0, int column = 0) const;
    GlobalPoint globalPosition(const DetId& id, const LocalPoint& local_point) const;

    // zside returns +/- 1
    int zside(const DetId& id) const;

    unsigned int layer(const DetId&) const;
    int module(const DetId&) const;
    std::pair<float, float> pixelInModule(const DetId& id, const int row, const int column) const;
    std::pair<uint8_t, uint8_t> pixelInModule(const DetId& id, const LocalPoint& local_point) const;
    int crystalInModule(const DetId&) const;

    // 4-vector helper functions using GlobalPoint
    float eta(const GlobalPoint& position, const float& vertex_z = 0.) const;
    float phi(const GlobalPoint& position) const;
    float pt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z = 0.) const;

    // 4-vector helper functions using DetId
    float eta(const DetId& id, const LocalPoint& local_point, const float& vertex_z = 0.) const;
    float phi(const DetId& id, const LocalPoint& local_point) const;
    float pt(const DetId& id, const LocalPoint& local_point, const float& hitEnergy, const float& vertex_z = 0.) const;

    inline const MTDGeometry* geometry() const { return geom_; };
    inline const MTDTopology* topology() const { return topology_; };

  private:
    const MTDGeometry* geom_;
    const MTDTopology* topology_;
  };
}  // namespace mtd

#endif
