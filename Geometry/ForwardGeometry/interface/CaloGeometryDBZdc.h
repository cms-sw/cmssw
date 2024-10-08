#ifndef Geometry_ForwardGeometry_CaloGeometryDBZdc_h
#define Geometry_ForwardGeometry_CaloGeometryDBZdc_h

#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"

namespace calogeometryDBEPimpl {
  template <>
  struct AdditionalTokens<ZdcGeometry> {
    void makeTokens(edm::ESConsumesCollectorT<ZdcGeometry::AlignedRecord>& cc) {
      topology = cc.consumesFrom<ZdcTopology, HcalRecNumberingRecord>(edm::ESInputTag{});
    }
    edm::ESGetToken<ZdcTopology, HcalRecNumberingRecord> topology;
  };
}  // namespace calogeometryDBEPimpl

#endif
