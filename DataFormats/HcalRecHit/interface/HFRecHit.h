#ifndef HFRECHIT_H
#define HFRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

namespace cms {

  /** \class HFRecHit
      
  $Date: $
  $Revision: $
  \author J. Mans - Minnesota
  */
  class HFRecHit : public CaloRecHit {
  public:
    HFRecHit();
    HFRecHit(const HcalDetId& id, float energy, float time);
    /// get the id
    const HcalDetId& id() const { return id_; }
    /// get the id for generic use
    virtual DetId genericId() const;
  private:
    HcalDetId id_;
  };

  std::ostream& operator<<(std::ostream& s, const HFRecHit& hit);
}

#endif
