#ifndef DATAFORMATS_HCALRECHIT_HORECHIT_H
#define DATAFORMATS_HCALRECHIT_HORECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


/** \class HORecHit
 *    
 * \author J. Mans - Minnesota
 */
class HORecHit : public CaloRecHit {
public:
  typedef HcalDetId key_type;

  constexpr HORecHit() : CaloRecHit() {}

  constexpr HORecHit(const HcalDetId& id, float energy, float time) 
      : CaloRecHit(id,energy,time) {}
  /// get the id
  constexpr HcalDetId id() const { return HcalDetId(detid()); }
};

std::ostream& operator<<(std::ostream& s, const HORecHit& hit);


#endif
