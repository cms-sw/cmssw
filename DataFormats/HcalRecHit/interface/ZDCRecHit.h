#ifndef DATAFORMATS_HCALRECHIT_ZDCRECHIT_H
#define DATAFORMATS_HCALRECHIT_ZDCRECHIT_H 1

#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"


/** \class ZDCRecHit
 *  
 * $Date: 2011/09/15 09:25:06 $
 * $Revision: 1.2 $
 *\author J. Mans - Minnesota
 */
class ZDCRecHit : public CaloRecHit {
public:
  typedef HcalZDCDetId key_type;

  ZDCRecHit();
  ZDCRecHit(const HcalZDCDetId& id, float energy, float time, float lowGainEnergy);
  /// get the id
  HcalZDCDetId id() const { return HcalZDCDetId(detid()); }
  // follow EcalRecHit method of adding variable flagBits_ to CaloRecHit
  float lowGainEnergy() const { return lowGainEnergy_;};
private:
  float lowGainEnergy_;
};

std::ostream& operator<<(std::ostream& s, const ZDCRecHit& hit);

#endif
