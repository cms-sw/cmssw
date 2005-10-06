#ifndef GEOMETRY_CALOTOPOLOGY_CALOTOWERTOPOLOGY_H
#define GEOMETRY_CALOTOPOLOGY_CALOTOWERTOPOLOGY_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include <vector>

/** \class CaloTowerTopology
  *  
  * $Date: 2005/10/05 21:42:39 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class CaloTowerTopology {
public:
  CaloTowerTopology();

  /// Get the tower id for this det id (or null if not known)
  CaloTowerDetId towerOf(const DetId& id) const;

  /// Get the constituent detids for this tower id ( not yet implemented )
  std::vector<DetId> constituentsOf(const CaloTowerDetId& id) const;
};

#endif
