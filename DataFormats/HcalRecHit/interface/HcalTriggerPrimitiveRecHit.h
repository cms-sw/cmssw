#ifndef HCALTRIGGERPRIMITIVERECHIT_H
#define HCALTRIGGERPRIMITIVERECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

namespace cms {

  /** \class HcalTriggerPrimitiveRecHit
      
  $Date: $
  $Revision: $
  \author J. Mans - Minnesota
  */
  class HcalTriggerPrimitiveRecHit : public CaloRecHit {
  public:
    HcalTriggerPrimitiveRecHit();
    explicit HcalTriggerPrimitiveRecHit(const HcalTrigTowerDetId& id, float energy, float time, int bunch=0, int index=0, int n=1);

    /// get the id
    const HcalTrigTowerDetId& id() const { return id_; }
    /// get the id for generic use
    virtual DetId genericId() const;
    /// get the number of trigger primitive rec hits in this event for this tower
    int towerCount() const { return count_; }
    /// get the index of this trigger primitive rec hit in this event
    int index() const { return index_; }
    /// get the relative bunch number of this trigger primitive rec hit
    int bunchRelative() const { return bunch_; }

  private:
    HcalTrigTowerDetId id_;
    char bunch_;
    char index_;
    char count_;
  };

  std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveRecHit& hit);
}

#endif
