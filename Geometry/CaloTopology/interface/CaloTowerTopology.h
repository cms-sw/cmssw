#ifndef GEOMETRY_CALOTOPOLOGY_CALOTOWERTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_CALOTOWERTOPOLOGY_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

/** \class CaloTowerTopology
  *  
  * $Date: $
  * $Revision: $
  * \author J. Mans - Minnesota
  */
class CaloTowerTopology {
public:
  CaloTowerTopology();

  CaloTowerDetId towerOf(const DetId& id) const;

  std::vector<DetId> constituentsOf(const CaloTowerDetId& id) const;
};

#endif
