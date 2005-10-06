#ifndef GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H
#define GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H 1

#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

/** \class CaloTowerHardcodeGeometryLoader
  *  
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class CaloTowerHardcodeGeometryLoader {
public:
  std::auto_ptr<CaloSubdetectorGeometry> load();
private:
  const CaloCellGeometry* makeCell(int ieta, int iphi) const;
  HcalTopology limits; // just for the ring limits

};

#endif
