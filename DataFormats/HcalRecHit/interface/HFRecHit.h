#ifndef DATAFORMATS_HCALRECHIT_HFRECHIT_H
#define DATAFORMATS_HCALRECHIT_HFRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


/** \class HFRecHit
 *  
 * $Date: 2005/09/26 14:10:28 $
 * $Revision: 1.3 $
 *\author J. Mans - Minnesota
 */
class HFRecHit : public CaloRecHit {
public:
  typedef HcalDetId key_type;

  HFRecHit();
  HFRecHit(const HcalDetId& id, float energy, float time);
  /// get the id
  HcalDetId id() const { return HcalDetId(detid()); }
};

std::ostream& operator<<(std::ostream& s, const HFRecHit& hit);

#endif
