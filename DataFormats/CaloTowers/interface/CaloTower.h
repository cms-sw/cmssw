#ifndef DATAFORMATS_CALOTOWERS_CALOTOWER_H
#define DATAFORMATS_CALOTOWERS_CALOTOWER_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include <vector>
#include <math.h>

/** \struct CaloTower
    
$Date: 2005/09/15 14:42:57 $
$Revision: 1.1 $
\author J. Mans - Minnesota
*/
struct CaloTower {
  typedef CaloTowerDetId key_type; // for SortedCollection

  CaloTowerDetId id_;
  double eT;
  double eT_em, eT_had, eT_outer;
  double eta, phi;
  std::vector<DetId> constituents;
  
  CaloTower(); // for persistence
  CaloTower(const CaloTowerDetId& id);
  double e() const { return eT*cosh(eta); }
  double e_em() const { return eT_em*cosh(eta); }
  double e_had() const { return eT_had*cosh(eta); }
  double e_outer() const { return eT_outer*cosh(eta); }
  CaloTowerDetId id() const { return id_; } // needed for SortedCollection
};

std::ostream& operator<<(std::ostream& s, const CaloTower& ct);

#endif
