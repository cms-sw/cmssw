#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include <iostream>




int main() {

  HcalHardcodeGeometryLoader l;
  std::auto_ptr<CaloSubdetectorGeometry> b=l.load(DetId::Hcal,HcalBarrel);
  std::auto_ptr<CaloSubdetectorGeometry> e=l.load(DetId::Hcal,HcalEndcap);
  std::auto_ptr<CaloSubdetectorGeometry> o=l.load(DetId::Hcal,HcalOuter);
  std::auto_ptr<CaloSubdetectorGeometry> f=l.load(DetId::Hcal,HcalForward);

  std::vector<DetId> ids=b->getValidDetIds(DetId::Hcal,HcalBarrel);
  for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) {
    HcalDetId hid=(*i);
    if (hid.iphi()!=1) continue;
    const CaloCellGeometry* geom=b->getGeometry(hid);
    std::vector<GlobalPoint> corners=geom->getCorners();
    std::cout << hid << std::endl;
    for (std::vector<GlobalPoint>::iterator j=corners.begin(); j!=corners.end(); j++) {
      std::cout << "  " << *j << std::endl;
    }
  }

  std::cout << std::endl << " FORWARD : " << std::endl;
  ids=f->getValidDetIds(DetId::Hcal,HcalForward);
  for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) {
    HcalDetId hid=(*i);
    if (hid.iphi()!=1 && hid.iphi()!=2 && hid.iphi()!=3) continue;
    std::cout << hid << std::endl;
    
    const CaloCellGeometry* geom=f->getGeometry(hid);
    std::vector<GlobalPoint> corners=geom->getCorners();
    for (std::vector<GlobalPoint>::iterator j=corners.begin(); j!=corners.end(); j++) {
      std::cout << "  " << *j << std::endl;
    }
  }


  std::cout << std::endl << " ENDCAP : " << std::endl;
  ids=e->getValidDetIds(DetId::Hcal,HcalEndcap);
  for (std::vector<DetId>::iterator i=ids.begin(); i!=ids.end(); i++) {
    HcalDetId hid=(*i);
    if (hid.iphi()!=1 && hid.iphi()!=2 && hid.iphi()!=3) continue;
    std::cout << hid << std::endl;
    
    const CaloCellGeometry* geom=e->getGeometry(hid);
    std::vector<GlobalPoint> corners=geom->getCorners();
    for (std::vector<GlobalPoint>::iterator j=corners.begin(); j!=corners.end(); j++) {
      std::cout << "  " << *j << std::endl;
    }
  }

  return 0;
}
