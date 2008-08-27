#ifndef GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H
#define GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H 1

#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

/** \class CaloTowerHardcodeGeometryLoader
  *  
  * $Date: 2007/09/07 22:05:51 $
  * $Revision: 1.2 $
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
