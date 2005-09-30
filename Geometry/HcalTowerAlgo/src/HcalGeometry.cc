#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometryLoader.h"

namespace cms {

HcalGeometry::HcalGeometry() {
  HcalGeometryLoader loader;
  loader.fill(validIds_, cellGeometries_);
}

}

