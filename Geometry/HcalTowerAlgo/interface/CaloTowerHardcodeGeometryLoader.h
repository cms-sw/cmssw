#ifndef GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H
#define GEOMETRY_HCALTOWERALGO_CALOTOWERHARDCODEGEOMETRYLOADER_H 1

#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include <memory>

/** \class CaloTowerHardcodeGeometryLoader
  *  
  * $Date: 2012/08/15 14:52:43 $
  * $Revision: 1.6 $
  * \author J. Mans - Minnesota
  */
class CaloTowerHardcodeGeometryLoader {
public:
  std::auto_ptr<CaloSubdetectorGeometry> load(const HcalTopology *limits);
private:
  void makeCell(int ieta, int iphi, CaloSubdetectorGeometry* geom) const;
  const HcalTopology *m_limits; // just for the ring limits

};

#endif
