#ifndef HBHERECHIT_H
#define HBHERECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

namespace cms {

  /** \class HBHERecHit
      
  $Date: 2005/07/26 15:52:19 $
  $Revision: 1.1 $
  \author J. Mans - Minnesota
  */
  class HBHERecHit : public CaloRecHit {
  public:
    HBHERecHit();
    HBHERecHit(const HcalDetId& id, float energy, float time);
    /// get the id
    HcalDetId hcal_id() const { return HcalDetId(id()); }
  };

  std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit);
}

#endif
