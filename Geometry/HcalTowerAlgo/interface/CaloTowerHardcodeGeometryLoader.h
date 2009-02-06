#ifndef GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H
#define GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H 1

#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include <memory>

/** \class CaloTowerHardcodeGeometryLoader
  *  
  * $Date: 2008/08/27 15:57:15 $
  * $Revision: 1.3 $
  * \author J. Mans - Minnesota
  */
class CaloTowerHardcodeGeometryLoader {
public:
  std::auto_ptr<CaloSubdetectorGeometry> load();
private:
  CaloCellGeometry* makeCell(int ieta, int iphi, CaloSubdetectorGeometry* geom) const;
  HcalTopology limits; // just for the ring limits

};

#endif
