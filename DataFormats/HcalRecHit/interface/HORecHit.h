#ifndef HORECHIT_H
#define HORECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

namespace cms {

  /** \class HORecHit
      
  $Date: $
  $Revision: $
  \author J. Mans - Minnesota
  */
  class HORecHit : public CaloRecHit {
  public:
    HORecHit();
    HORecHit(const HcalDetId& id, float energy, float time);
    /// get the id
    const HcalDetId& id() const { return id_; }
    /// get the id for generic use
    virtual DetId genericId() const;
  private:
    HcalDetId id_;
  };

  std::ostream& operator<<(std::ostream& s, const HORecHit& hit);
}

#endif
