#ifndef HFRECHIT_H
#define HFRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

namespace cms {

  /** \class HFRecHit
      
  $Date: 2005/09/15 14:44:25 $
  $Revision: 1.2 $
  \author J. Mans - Minnesota
  */
  class HFRecHit : public CaloRecHit {
  public:
    HFRecHit();
    HFRecHit(const HcalDetId& id, float energy, float time);
    /// get the id
    HcalDetId id() const { return HcalDetId(detid()); }
  };

  std::ostream& operator<<(std::ostream& s, const HFRecHit& hit);
}

#endif
