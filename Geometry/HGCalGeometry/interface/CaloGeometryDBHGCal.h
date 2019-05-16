#ifndef Geometry_HGCalGeometry_CaloGeometryDBHGCal_h
#define Geometry_HGCalGeometry_CaloGeometryDBHGCal_h

#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

// Specializations for HGCal
namespace calogeometryDBEPimpl {
  static constexpr auto nameHGCal = "HGCalEESensitive";
  template <>
  struct GeometryTraits<HGCalGeometry, true> {
    using TokenType = edm::ESGetToken<HGCalGeometry, IdealGeometryRecord>;

    static TokenType makeToken(edm::ESConsumesCollectorT<HGCalGeometry::AlignedRecord>& cc) {
      return cc.template consumesFrom<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameHGCal});
    }
  };
  template <>
  struct AdditionalTokens<HGCalGeometry> {
    void makeTokens(edm::ESConsumesCollectorT<HGCalGeometry::AlignedRecord>& cc) {
      topology = cc.consumesFrom<HGCalTopology, IdealGeometryRecord>(edm::ESInputTag{"", nameHGCal});
    }
    edm::ESGetToken<HGCalTopology, IdealGeometryRecord> topology;
  };
}

#endif
