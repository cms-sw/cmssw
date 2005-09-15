#ifndef HFRECHIT_H
#define HFRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

namespace cms {

  /** \class HFRecHit
      
  $Date: 2005/07/26 15:52:19 $
  $Revision: 1.1 $
  \author J. Mans - Minnesota
  */
  class HFRecHit : public CaloRecHit {
  public:
    HFRecHit();
    HFRecHit(const HcalDetId& id, float energy, float time);
    /// get the id
    HcalDetId hcal_id() const { return HcalDetId(id()); }
  };

  std::ostream& operator<<(std::ostream& s, const HFRecHit& hit);
}

#endif
