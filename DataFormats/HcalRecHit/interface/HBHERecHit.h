#ifndef HBHERECHIT_H
#define HBHERECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

namespace cms {

  /** \class HBHERecHit
      
  $Date: $
  $Revision: $
  \author J. Mans - Minnesota
  */
  class HBHERecHit : public CaloRecHit {
  public:
    HBHERecHit();
    HBHERecHit(const HcalDetId& id, float energy, float time);
    /// get the id
    const HcalDetId& id() const { return id_; }
    /// get the id for generic use
    virtual DetId genericId() const;
  private:
    HcalDetId id_;
  };

  std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit);
}

#endif
