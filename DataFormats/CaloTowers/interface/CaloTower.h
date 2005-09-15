#ifndef DATAFORMATS_CALOTOWERS_CALOTOWER_H
#define DATAFORMATS_CALOTOWERS_CALOTOWER_H 1

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include <vector>
#include <math.h>

namespace cms {

  /** \struct CaloTower
    
  $Date: $
  $Revision: $
  \author J. Mans - Minnesota
  */
  struct CaloTower {
    CaloTowerDetId id;
    double eT;
    double eT_em, eT_had, eT_outer;
    double eta, phi;
    std::vector<DetId> constituents;

    double e() const { return eT*cosh(eta); }
    double e_em() const { return eT_em*cosh(eta); }
    double e_had() const { return eT_had*cosh(eta); }
    double e_outer() const { return eT_outer*cosh(eta); }
  };
}

#endif
