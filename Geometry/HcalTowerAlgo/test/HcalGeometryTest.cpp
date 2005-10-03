#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include <iostream>


int main() {

  HcalHardcodeGeometryLoader l;
  l.load(DetId::Hcal,HcalBarrel);
  l.load(DetId::Hcal,HcalEndcap);
  l.load(DetId::Hcal,HcalOuter);
  l.load(DetId::Hcal,HcalForward);

  return 0;
}
