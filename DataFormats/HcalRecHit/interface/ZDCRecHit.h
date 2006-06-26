#ifndef DATAFORMATS_HCALRECHIT_ZDCRECHIT_H
#define DATAFORMATS_HCALRECHIT_ZDCRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


/** \class ZDCRecHit
 *  
 * $Date: 2005/10/04 20:33:53 $
 * $Revision: 1.4 $
 *\author J. Mans - Minnesota
 */
class ZDCRecHit : public CaloRecHit {
public:
  typedef HcalZDCDetId key_type;

  ZDCRecHit();
  ZDCRecHit(const HcalZDCDetId& id, float energy, float time);
  /// get the id
  HcalZDCDetId id() const { return HcalZDCDetId(detid()); }
};

std::ostream& operator<<(std::ostream& s, const ZDCRecHit& hit);

#endif
