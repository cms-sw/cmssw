#ifndef DATAFORMATS_HCALRECHIT_HORECHIT_H
#define DATAFORMATS_HCALRECHIT_HORECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


/** \class HORecHit
 *    
 * $Date: 2005/10/04 20:33:53 $
 * $Revision: 1.4 $
 * \author J. Mans - Minnesota
 */
class HORecHit : public CaloRecHit {
public:
  typedef HcalDetId key_type;

  HORecHit();
  HORecHit(const HcalDetId& id, float energy, float time);
  /// get the id
  HcalDetId id() const { return HcalDetId(detid()); }
};

std::ostream& operator<<(std::ostream& s, const HORecHit& hit);


#endif
