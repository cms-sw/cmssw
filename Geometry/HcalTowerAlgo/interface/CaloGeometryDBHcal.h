#ifndef Geometry_HcalTowerAlgo_CaloGeometryDBHcal_h
#define Geometry_HcalTowerAlgo_CaloGeometryDBHcal_h

#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

namespace calogeometryDBEPimpl {
  template <>
  struct AdditionalTokens<HcalGeometry> {
    void makeTokens(edm::ESConsumesCollectorT<HcalGeometry::AlignedRecord>& cc) {
      topology = cc.consumesFrom<HcalTopology, HcalRecNumberingRecord>(edm::ESInputTag{});
    }
    edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topology;
  };
}

#endif
